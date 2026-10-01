-- Animal Simulator - Adaptive Fireball Mobile v2
-- F4 = Toggle GUI | V = Toggle Auto-Fire | N = Cycle body part | F10 = Destroy
--
-- Important:
-- * COOLDOWN is exactly 1.5 seconds.
-- * Player:GetNetworkPing() is RTT in seconds, so ping ms = value * 1000.
-- * The game remote only receives (aimPosition, "NewFireball"). Projectile speed is NOT
--   controlled by this script. PROJECTILE_SPEED_SEED is only a prediction-model seed.
-- * The predictor keeps a separate movement profile for every target and learns from:
--   short/long movement trends, turning/juking, observed target speed, prediction residuals,
--   your ping, and (when detectable) the actual visible projectile speed.

-- ===== Prevent duplicates =====
if _G.FireballAdaptiveMobileV2 then
    local old = _G.FireballAdaptiveMobileV2
    for _, key in ipairs({"InputConn", "AutoConn", "ProjectileConn", "DragConn"}) do
        if old[key] then
            pcall(function() old[key]:Disconnect() end)
        end
    end
    if old.GUI then
        pcall(function() old.GUI:Destroy() end)
    end
    _G.FireballAdaptiveMobileV2 = nil
end

_G.FireballAdaptiveMobileV2 = {}
local STATE = _G.FireballAdaptiveMobileV2

-- ===== Services =====
local Players = game:GetService("Players")
local UserInputService = game:GetService("UserInputService")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Workspace = game:GetService("Workspace")

local LocalPlayer = Players.LocalPlayer

local skillsFolder = ReplicatedStorage:WaitForChild("SkillsInRS", 5)
if not skillsFolder then
    warn("[AdaptiveFireball] SkillsInRS not found.")
    return
end

local remote = skillsFolder:WaitForChild("RemoteEvent", 5)
if not remote then
    warn("[AdaptiveFireball] SkillsInRS.RemoteEvent not found.")
    return
end

-- ===== Config =====
local COOLDOWN = 1.5
local MAX_DISTANCE = 150

-- This is NOT claimed to be Animal Simulator's real internal speed.
-- It is only the starting value for the intercept model.
-- If a visible fireball BasePart can be detected, the script self-calibrates this value.
local PROJECTILE_SPEED_SEED = 220
local MIN_PROJECTILE_SPEED = 60
local MAX_PROJECTILE_SPEED = 450

-- GetNetworkPing() is RTT. We start below full RTT because the exact server/replication
-- timing of Animal Simulator is not public. Per-target residual learning corrects this.
local NETWORK_RTT_FACTOR = 0.65

local SAMPLE_INTERVAL = 1 / 30
local HISTORY_SECONDS = 1.15
local SHORT_WINDOW = 0.16
local LONG_WINDOW = 0.50
local MAX_MODEL_HORIZON = 1.35
local MAX_PLAUSIBLE_TARGET_SPEED = 260
local MAX_ACCEL = 260

local BODY_PARTS = {
    {name = "Right Foot", part = "RightFoot", fallback = "HumanoidRootPart"},
    {name = "Left Foot",  part = "LeftFoot",  fallback = "HumanoidRootPart"},
    {name = "Torso",      part = "HumanoidRootPart", fallback = "HumanoidRootPart"},
    {name = "UpperTorso", part = "UpperTorso", fallback = "HumanoidRootPart"},
    {name = "Head",       part = "Head", fallback = "HumanoidRootPart"},
}

local currentBodyPartIndex = 1
local autoFire = false
local targetName = ""
local lastFire = -math.huge

local projectileSpeedModel = PROJECTILE_SPEED_SEED
local smoothedPingMs = nil
local lastShotProbe = nil

-- userId -> adaptive movement profile
local profiles = {}

-- ===== Small math helpers =====
local ZERO = Vector3.new(0, 0, 0)

local function clampMagnitude(v, maxMag)
    local m = v.Magnitude
    if m > maxMag and m > 0 then
        return v.Unit * maxMag
    end
    return v
end

local function expAlpha(dt, tau)
    if tau <= 0 then
        return 1
    end
    return 1 - math.exp(-dt / tau)
end

local function safeUnit(v)
    local m = v.Magnitude
    if m < 1e-5 then
        return ZERO
    end
    return v / m
end

local function dotUnit(a, b)
    local ua = safeUnit(a)
    local ub = safeUnit(b)
    if ua.Magnitude == 0 or ub.Magnitude == 0 then
        return 1
    end
    return math.clamp(ua:Dot(ub), -1, 1)
end

-- ===== Character helpers =====
local function isAlive(char)
    if not char then return false end
    local hum = char:FindFirstChildOfClass("Humanoid")
    return hum ~= nil and hum.Health > 0
end

local function findPlayerByPartial(name)
    if not name or name == "" then return nil end
    local needle = string.lower(name)

    -- Exact username/display name first.
    for _, p in ipairs(Players:GetPlayers()) do
        if p ~= LocalPlayer then
            if string.lower(p.Name) == needle or string.lower(p.DisplayName) == needle then
                return p
            end
        end
    end

    -- Then partial username/display name.
    for _, p in ipairs(Players:GetPlayers()) do
        if p ~= LocalPlayer then
            if string.find(string.lower(p.Name), needle, 1, true)
                or string.find(string.lower(p.DisplayName), needle, 1, true) then
                return p
            end
        end
    end

    return nil
end

local function getHRP(char)
    return char and char:FindFirstChild("HumanoidRootPart")
end

local function getTargetPart(char)
    if not char then return nil end
    local cfg = BODY_PARTS[currentBodyPartIndex]
    return char:FindFirstChild(cfg.part)
        or char:FindFirstChild(cfg.fallback)
        or char:FindFirstChild("HumanoidRootPart")
end

local function belongsToCharacter(part)
    local node = part
    for _ = 1, 5 do
        if not node then break end
        if node:IsA("Model") and node:FindFirstChildOfClass("Humanoid") then
            return true
        end
        node = node.Parent
    end
    return false
end

-- ===== Ping =====
local function getPingMs()
    local ok, seconds = pcall(function()
        return LocalPlayer:GetNetworkPing()
    end)

    if not ok or type(seconds) ~= "number" then
        return smoothedPingMs or 0
    end

    local rawMs = math.max(0, seconds * 1000) -- RTT, not *2000

    if not smoothedPingMs then
        smoothedPingMs = rawMs
    else
        -- Smooth ping spikes without making it slow to react.
        smoothedPingMs = smoothedPingMs + (rawMs - smoothedPingMs) * 0.20
    end

    return smoothedPingMs
end

-- ===== Per-player movement learning =====
local function newProfile()
    return {
        samples = {},
        lastSampleT = nil,

        fastVel = ZERO,
        slowVel = ZERO,
        previousFastVel = ZERO,
        accel = ZERO,

        speedEMA = 0,
        peakSpeed = 0,
        turnRateEMA = 0,
        jukeEMA = 0,

        -- Online correction learned separately for every target.
        leadBiasSeconds = 0,
        lateralBias = ZERO,

        shotsLearned = 0,
    }
end

local function getProfile(player)
    local id = player.UserId
    if not profiles[id] then
        profiles[id] = newProfile()
    end
    return profiles[id]
end

local function regressionVelocity(samples, windowSeconds)
    local n = #samples
    if n < 2 then return ZERO end

    local now = samples[n].t
    local first = n

    while first > 1 and (now - samples[first - 1].t) <= windowSeconds do
        first -= 1
    end

    local count = n - first + 1
    if count < 2 then return ZERO end

    local meanT = 0
    local meanP = ZERO
    local weightSum = 0

    -- Slightly favor newer samples.
    for i = first, n do
        local age = now - samples[i].t
        local w = math.exp(-age / math.max(windowSeconds, 0.001))
        meanT += samples[i].t * w
        meanP += samples[i].pos * w
        weightSum += w
    end

    if weightSum <= 0 then return ZERO end

    meanT /= weightSum
    meanP /= weightSum

    local numerator = ZERO
    local denominator = 0

    for i = first, n do
        local age = now - samples[i].t
        local w = math.exp(-age / math.max(windowSeconds, 0.001))
        local dt = samples[i].t - meanT
        numerator += (samples[i].pos - meanP) * (dt * w)
        denominator += dt * dt * w
    end

    if denominator < 1e-6 then return ZERO end
    return numerator / denominator
end

local function updateProfile(player, hrp)
    local p = getProfile(player)
    local now = os.clock()

    if p.lastSampleT and (now - p.lastSampleT) < SAMPLE_INTERVAL then
        return p
    end

    local dt = p.lastSampleT and (now - p.lastSampleT) or SAMPLE_INTERVAL
    p.lastSampleT = now

    table.insert(p.samples, {
        t = now,
        pos = hrp.Position,
    })

    while #p.samples > 2 and (now - p.samples[1].t) > HISTORY_SECONDS do
        table.remove(p.samples, 1)
    end

    if #p.samples < 3 then
        return p
    end

    local fast = regressionVelocity(p.samples, SHORT_WINDOW)
    local slow = regressionVelocity(p.samples, LONG_WINDOW)

    -- Reject obvious teleport/replication spikes instead of poisoning the profile.
    if fast.Magnitude > MAX_PLAUSIBLE_TARGET_SPEED then
        fast = slow
    end
    if slow.Magnitude > MAX_PLAUSIBLE_TARGET_SPEED then
        slow = clampMagnitude(slow, MAX_PLAUSIBLE_TARGET_SPEED)
    end

    p.previousFastVel = p.fastVel
    p.fastVel = fast
    p.slowVel = slow

    local speed = fast.Magnitude
    local speedAlpha = expAlpha(dt, 0.20)
    p.speedEMA += (speed - p.speedEMA) * speedAlpha

    -- Peak adapts upward immediately and decays slowly.
    p.peakSpeed = math.max(speed, p.peakSpeed * math.exp(-dt / 5.0))

    local accelEstimate = (fast - slow) / math.max((LONG_WINDOW - SHORT_WINDOW) * 0.5, 0.05)
    p.accel = clampMagnitude(
        p.accel:Lerp(accelEstimate, expAlpha(dt, 0.16)),
        MAX_ACCEL
    )

    local turnRate = 0
    if p.previousFastVel.Magnitude > 3 and fast.Magnitude > 3 then
        local d = math.clamp(dotUnit(p.previousFastVel, fast), -1, 1)
        local angle = math.acos(d)
        turnRate = angle / math.max(dt, 1 / 120)
    end
    p.turnRateEMA += (turnRate - p.turnRateEMA) * expAlpha(dt, 0.16)

    -- "Juke" = disagreement between immediate and longer-term movement + turning.
    -- This adapts to players who spam direction changes instead of assuming everyone runs straight.
    local dirMismatch = (1 - dotUnit(fast, slow)) * 0.5
    local velocityMismatch = (fast - slow).Magnitude / math.max(speed, 12)
    local turnFactor = math.clamp(p.turnRateEMA / 7.0, 0, 1)

    local rawJuke = math.clamp(
        dirMismatch * 0.45
        + math.clamp(velocityMismatch, 0, 1) * 0.35
        + turnFactor * 0.20,
        0,
        1
    )

    p.jukeEMA += (rawJuke - p.jukeEMA) * expAlpha(dt, 0.22)

    return p
end

local function predictHRP(profile, currentPos, horizon)
    horizon = math.clamp(horizon, 0, MAX_MODEL_HORIZON)

    local juke = math.clamp(profile.jukeEMA, 0, 1)

    -- Stable player: trust immediate movement + acceleration more.
    -- Juking player: blend more toward their longer-term movement habit.
    local velocityBlend = 0.18 + juke * 0.62
    local v = profile.fastVel:Lerp(profile.slowVel, velocityBlend)

    local accelWeight = (1 - juke) ^ 2
    local accelTime = math.min(horizon, 0.65)

    local predicted = currentPos
        + v * horizon
        + profile.accel * (0.5 * accelTime * accelTime * accelWeight)

    -- Small learned lateral correction for players with a repeated strafe/circle habit.
    predicted += profile.lateralBias * math.clamp(horizon / 0.40, 0, 1)

    -- Dynamic sanity envelope based on THIS player's actually observed speed.
    local learnedSpeedEnvelope = math.max(
        18,
        profile.speedEMA * 1.55,
        profile.peakSpeed * 1.25
    )

    local displacement = predicted - currentPos
    local maxDisplacement = learnedSpeedEnvelope * horizon + 4

    if displacement.Magnitude > maxDisplacement and displacement.Magnitude > 0 then
        predicted = currentPos + displacement.Unit * maxDisplacement
    end

    return predicted
end

-- ===== Intercept calculation =====
local function calculateAim(origin, localVelocity, targetHRP, targetPart, profile, pingMs)
    local pingSeconds = pingMs / 1000

    -- FireServer reaches the server after roughly part of the RTT. We also need to account
    -- for the fact that the target snapshot we see is not perfectly "now".
    -- Exact Animal Simulator replication timing is not public, so residual learning below
    -- tunes this per opponent instead of pretending one fixed ping multiplier is perfect.
    local networkLead = pingSeconds * NETWORK_RTT_FACTOR + profile.leadBiasSeconds
    networkLead = math.clamp(networkLead, 0, 0.38)

    -- Approximate where our own server-side character will be when the shot is processed.
    local outbound = math.clamp(pingSeconds * 0.5, 0, 0.18)
    local serverOrigin = origin + localVelocity * outbound

    local speed = math.clamp(projectileSpeedModel, MIN_PROJECTILE_SPEED, MAX_PROJECTILE_SPEED)

    local partOffset = targetPart.Position - targetHRP.Position
    local travelTime = (targetHRP.Position - serverOrigin).Magnitude / speed

    local predictedHRP = targetHRP.Position
    local horizon = networkLead + travelTime

    -- Iterate because target movement changes the travel distance, which changes arrival time.
    for _ = 1, 4 do
        horizon = math.clamp(networkLead + travelTime, 0, MAX_MODEL_HORIZON)
        predictedHRP = predictHRP(profile, targetHRP.Position, horizon)
        travelTime = (predictedHRP - serverOrigin).Magnitude / speed
    end

    horizon = math.clamp(networkLead + travelTime, 0, MAX_MODEL_HORIZON)
    predictedHRP = predictHRP(profile, targetHRP.Position, horizon)

    local aimPos = predictedHRP + partOffset

    return aimPos, predictedHRP, horizon, travelTime, networkLead, serverOrigin
end

-- ===== Online residual learning =====
local function scheduleResidualLearning(player, profile, predictedHRP, expectedHorizon, shotVelocity, shotJuke)
    local confidence = math.clamp(1 - shotJuke, 0.15, 1)
    local velocityAtShot = shotVelocity
    local speedAtShot = velocityAtShot.Magnitude

    task.delay(expectedHorizon + 0.03, function()
        if not profiles[player.UserId] then return end
        if not player.Character or not isAlive(player.Character) then return end

        local hrp = getHRP(player.Character)
        if not hrp then return end

        local errorVector = hrp.Position - predictedHRP

        -- Ignore respawns/teleports/huge unrelated discontinuities.
        if errorVector.Magnitude > 55 then return end

        if speedAtShot > 5 then
            local moveDir = velocityAtShot.Unit
            local alongStuds = errorVector:Dot(moveDir)
            local timeError = alongStuds / speedAtShot

            -- If target consistently ends up farther along their path, increase lead.
            -- If we consistently over-lead, reduce it.
            local correction = math.clamp(timeError, -0.20, 0.20) * 0.10 * confidence
            profile.leadBiasSeconds = math.clamp(
                profile.leadBiasSeconds + correction,
                -0.20,
                0.28
            )

            -- Learn repeated perpendicular/circle-strafe bias without letting one juke dominate.
            local lateral = errorVector - moveDir * alongStuds
            lateral = clampMagnitude(lateral, 14)
            profile.lateralBias = profile.lateralBias:Lerp(
                lateral * 0.30,
                0.06 * confidence
            )
        end

        profile.shotsLearned += 1
    end)
end

-- ===== Optional visible-projectile speed self-calibration =====
-- This does NOT touch the projectile. It only watches for a new moving BasePart that appears
-- near our last shot and travels in the aimed direction. If Animal Simulator exposes the
-- visible fireball this way, the prediction model learns its actual observed speed.
local function considerProjectileCandidate(obj)
    local probe = lastShotProbe
    if not probe or probe.calibrated or os.clock() > probe.expires then return end
    if not obj:IsA("BasePart") then return end
    if belongsToCharacter(obj) then return end

    task.defer(function()
        if not obj.Parent then return end
        local p0 = obj.Position

        if (p0 - probe.origin).Magnitude > 35 then
            return
        end

        local t0 = os.clock()
        task.wait(0.055)

        if not obj.Parent then return end
        local dt = os.clock() - t0
        if dt <= 0 then return end

        local delta = obj.Position - p0
        local observedSpeed = delta.Magnitude / dt

        if observedSpeed < MIN_PROJECTILE_SPEED or observedSpeed > MAX_PROJECTILE_SPEED then
            return
        end

        if delta.Magnitude > 0.05 then
            local directionMatch = delta.Unit:Dot(probe.direction)
            if directionMatch < 0.70 then
                return
            end
        end

        probe.calibrated = true
        projectileSpeedModel += (observedSpeed - projectileSpeedModel) * 0.28

        print(string.format(
            "[AdaptiveFireball] Projectile model calibrated: %.1f studs/s (sample %.1f)",
            projectileSpeedModel,
            observedSpeed
        ))
    end)
end

STATE.ProjectileConn = Workspace.DescendantAdded:Connect(considerProjectileCandidate)

-- ===== Mobile-friendly GUI =====
local gui = Instance.new("ScreenGui")
gui.Name = "FireballAdaptiveMobileV2GUI"
gui.ResetOnSpawn = false
gui.IgnoreGuiInset = false
gui.ZIndexBehavior = Enum.ZIndexBehavior.Sibling
gui.Parent = LocalPlayer:WaitForChild("PlayerGui")
STATE.GUI = gui

-- Small always-available button for mobile.
local miniToggle = Instance.new("TextButton")
miniToggle.Name = "MiniToggle"
miniToggle.Size = UDim2.new(0, 92, 0, 44)
miniToggle.Position = UDim2.new(0, 12, 0.45, 0)
miniToggle.BackgroundColor3 = Color3.fromRGB(25, 25, 25)
miniToggle.TextColor3 = Color3.fromRGB(255, 115, 115)
miniToggle.Text = "FIRE AIM"
miniToggle.Font = Enum.Font.SourceSansBold
miniToggle.TextSize = 15
miniToggle.AutoButtonColor = true
miniToggle.Active = true
miniToggle.Parent = gui

local miniCorner = Instance.new("UICorner")
miniCorner.CornerRadius = UDim.new(0, 10)
miniCorner.Parent = miniToggle

local miniStroke = Instance.new("UIStroke")
miniStroke.Thickness = 1
miniStroke.Transparency = 0.25
miniStroke.Color = Color3.fromRGB(255, 90, 90)
miniStroke.Parent = miniToggle

local frame = Instance.new("Frame")
frame.Name = "Main"
frame.Size = UDim2.new(0.92, 0, 0, 342)
frame.Position = UDim2.new(0.5, 0, 0.18, 0)
frame.AnchorPoint = Vector2.new(0.5, 0)
frame.BackgroundColor3 = Color3.fromRGB(18, 18, 18)
frame.BorderSizePixel = 0
frame.Visible = false
frame.Active = true
frame.Parent = gui

local sizeConstraint = Instance.new("UISizeConstraint")
sizeConstraint.MinSize = Vector2.new(300, 342)
sizeConstraint.MaxSize = Vector2.new(410, 342)
sizeConstraint.Parent = frame

local corner = Instance.new("UICorner")
corner.CornerRadius = UDim.new(0, 10)
corner.Parent = frame

local stroke = Instance.new("UIStroke")
stroke.Thickness = 1
stroke.Transparency = 0.40
stroke.Color = Color3.fromRGB(90, 90, 90)
stroke.Parent = frame

-- Drag handle / title.
local title = Instance.new("TextLabel")
title.Name = "DragHandle"
title.Size = UDim2.new(1, -54, 0, 34)
title.Position = UDim2.new(0, 10, 0, 4)
title.BackgroundTransparency = 1
title.Text = "Adaptive Fireball Mobile v2"
title.TextColor3 = Color3.fromRGB(255, 90, 90)
title.Font = Enum.Font.SourceSansBold
title.TextSize = 17
title.TextXAlignment = Enum.TextXAlignment.Left
title.Active = true
title.Parent = frame

local closeX = Instance.new("TextButton")
closeX.Name = "CloseX"
closeX.Size = UDim2.new(0, 40, 0, 32)
closeX.Position = UDim2.new(1, -46, 0, 5)
closeX.BackgroundColor3 = Color3.fromRGB(40, 40, 40)
closeX.Text = "X"
closeX.TextColor3 = Color3.fromRGB(240, 240, 240)
closeX.Font = Enum.Font.SourceSansBold
closeX.TextSize = 16
closeX.Parent = frame
local closeCorner = Instance.new("UICorner")
closeCorner.CornerRadius = UDim.new(0, 8)
closeCorner.Parent = closeX

local input = Instance.new("TextBox")
input.Name = "TargetInput"
input.Size = UDim2.new(1, -126, 0, 38)
input.Position = UDim2.new(0, 10, 0, 44)
input.PlaceholderText = "Username / display name..."
input.BackgroundColor3 = Color3.fromRGB(36, 36, 36)
input.TextColor3 = Color3.fromRGB(235, 235, 235)
input.PlaceholderColor3 = Color3.fromRGB(155, 155, 155)
input.Font = Enum.Font.SourceSans
input.TextSize = 15
input.ClearTextOnFocus = false
input.TextXAlignment = Enum.TextXAlignment.Left
input.Parent = frame
local inputCorner = Instance.new("UICorner")
inputCorner.CornerRadius = UDim.new(0, 8)
inputCorner.Parent = input

local lockButton = Instance.new("TextButton")
lockButton.Name = "LockTarget"
lockButton.Size = UDim2.new(0, 100, 0, 38)
lockButton.Position = UDim2.new(1, -110, 0, 44)
lockButton.BackgroundColor3 = Color3.fromRGB(55, 55, 55)
lockButton.TextColor3 = Color3.fromRGB(255, 255, 255)
lockButton.Text = "LOCK"
lockButton.Font = Enum.Font.SourceSansBold
lockButton.TextSize = 15
lockButton.Parent = frame
local lockCorner = Instance.new("UICorner")
lockCorner.CornerRadius = UDim.new(0, 8)
lockCorner.Parent = lockButton

local partLabel = Instance.new("TextLabel")
partLabel.Size = UDim2.new(1, -20, 0, 24)
partLabel.Position = UDim2.new(0, 10, 0, 88)
partLabel.BackgroundTransparency = 1
partLabel.TextColor3 = Color3.fromRGB(255, 130, 130)
partLabel.Font = Enum.Font.SourceSansBold
partLabel.TextSize = 14
partLabel.TextXAlignment = Enum.TextXAlignment.Left
partLabel.Parent = frame

local telemetry = Instance.new("TextLabel")
telemetry.Size = UDim2.new(1, -20, 0, 68)
telemetry.Position = UDim2.new(0, 10, 0, 114)
telemetry.BackgroundColor3 = Color3.fromRGB(27, 27, 27)
telemetry.BackgroundTransparency = 0.15
telemetry.TextColor3 = Color3.fromRGB(110, 255, 205)
telemetry.Font = Enum.Font.SourceSansBold
telemetry.TextSize = 13
telemetry.TextWrapped = true
telemetry.TextXAlignment = Enum.TextXAlignment.Left
telemetry.TextYAlignment = Enum.TextYAlignment.Center
telemetry.Text = "Distance: — | Target speed: — | Ping: —\nJuke: — | Lead: — | Projectile model: —"
telemetry.Parent = frame
local telemetryCorner = Instance.new("UICorner")
telemetryCorner.CornerRadius = UDim.new(0, 8)
telemetryCorner.Parent = telemetry

local autoButton = Instance.new("TextButton")
autoButton.Name = "AutoButton"
autoButton.Size = UDim2.new(0.5, -15, 0, 44)
autoButton.Position = UDim2.new(0, 10, 0, 190)
autoButton.BackgroundColor3 = Color3.fromRGB(62, 42, 42)
autoButton.TextColor3 = Color3.fromRGB(255, 150, 150)
autoButton.Text = "AUTO: OFF"
autoButton.Font = Enum.Font.SourceSansBold
autoButton.TextSize = 16
autoButton.Parent = frame
local autoCorner = Instance.new("UICorner")
autoCorner.CornerRadius = UDim.new(0, 8)
autoCorner.Parent = autoButton

local bodyButton = Instance.new("TextButton")
bodyButton.Name = "BodyButton"
bodyButton.Size = UDim2.new(0.5, -15, 0, 44)
bodyButton.Position = UDim2.new(0.5, 5, 0, 190)
bodyButton.BackgroundColor3 = Color3.fromRGB(42, 52, 62)
bodyButton.TextColor3 = Color3.fromRGB(175, 220, 255)
bodyButton.Text = "BODY: —"
bodyButton.Font = Enum.Font.SourceSansBold
bodyButton.TextSize = 15
bodyButton.Parent = frame
local bodyCorner = Instance.new("UICorner")
bodyCorner.CornerRadius = UDim.new(0, 8)
bodyCorner.Parent = bodyButton

local hideButton = Instance.new("TextButton")
hideButton.Name = "HideButton"
hideButton.Size = UDim2.new(0.5, -15, 0, 38)
hideButton.Position = UDim2.new(0, 10, 0, 242)
hideButton.BackgroundColor3 = Color3.fromRGB(48, 48, 48)
hideButton.TextColor3 = Color3.fromRGB(240, 240, 240)
hideButton.Text = "HIDE MENU"
hideButton.Font = Enum.Font.SourceSansBold
hideButton.TextSize = 14
hideButton.Parent = frame
local hideCorner = Instance.new("UICorner")
hideCorner.CornerRadius = UDim.new(0, 8)
hideCorner.Parent = hideButton

local destroyButton = Instance.new("TextButton")
destroyButton.Name = "DestroyButton"
destroyButton.Size = UDim2.new(0.5, -15, 0, 38)
destroyButton.Position = UDim2.new(0.5, 5, 0, 242)
destroyButton.BackgroundColor3 = Color3.fromRGB(75, 35, 35)
destroyButton.TextColor3 = Color3.fromRGB(255, 185, 185)
destroyButton.Text = "DESTROY"
destroyButton.Font = Enum.Font.SourceSansBold
destroyButton.TextSize = 14
destroyButton.Parent = frame
local destroyCorner = Instance.new("UICorner")
destroyCorner.CornerRadius = UDim.new(0, 8)
destroyCorner.Parent = destroyButton

local status = Instance.new("TextLabel")
status.Size = UDim2.new(1, -20, 0, 48)
status.Position = UDim2.new(0, 10, 0, 286)
status.BackgroundTransparency = 1
status.TextColor3 = Color3.fromRGB(120, 255, 120)
status.Font = Enum.Font.SourceSans
status.TextSize = 13
status.TextWrapped = true
status.TextXAlignment = Enum.TextXAlignment.Left
status.TextYAlignment = Enum.TextYAlignment.Top
status.Text = "Touch LOCK -> choose body -> AUTO ON\nCooldown: exactly 1.50s"
status.Parent = frame

local function updatePartLabel()
    local bodyName = BODY_PARTS[currentBodyPartIndex].name
    partLabel.Text = "Body Part: " .. bodyName
    bodyButton.Text = "BODY: " .. string.upper(bodyName)
end
updatePartLabel()

local function updateAutoButton()
    if autoFire then
        autoButton.Text = "AUTO: ON"
        autoButton.BackgroundColor3 = Color3.fromRGB(38, 72, 45)
        autoButton.TextColor3 = Color3.fromRGB(150, 255, 170)
    else
        autoButton.Text = "AUTO: OFF"
        autoButton.BackgroundColor3 = Color3.fromRGB(62, 42, 42)
        autoButton.TextColor3 = Color3.fromRGB(255, 150, 150)
    end
end
updateAutoButton()

-- Touch/mouse dragging, useful on phones where the default position covers gameplay.
do
    local dragging = false
    local dragStart = nil
    local startPos = nil
    local dragInput = nil

    title.InputBegan:Connect(function(inp)
        if inp.UserInputType == Enum.UserInputType.Touch
            or inp.UserInputType == Enum.UserInputType.MouseButton1 then
            dragging = true
            dragStart = inp.Position
            startPos = frame.Position

            inp.Changed:Connect(function()
                if inp.UserInputState == Enum.UserInputState.End then
                    dragging = false
                end
            end)
        end
    end)

    title.InputChanged:Connect(function(inp)
        if inp.UserInputType == Enum.UserInputType.Touch
            or inp.UserInputType == Enum.UserInputType.MouseMovement then
            dragInput = inp
        end
    end)

    STATE.DragConn = UserInputService.InputChanged:Connect(function(inp)
        if dragging and inp == dragInput and dragStart and startPos then
            local delta = inp.Position - dragStart
            frame.Position = UDim2.new(
                startPos.X.Scale,
                startPos.X.Offset + delta.X,
                startPos.Y.Scale,
                startPos.Y.Offset + delta.Y
            )
        end
    end)
end

-- ===== Fire =====
local function fireAtTarget()
    local now = os.clock()
    if now - lastFire < COOLDOWN then return end

    local targetPlayer = findPlayerByPartial(targetName)
    if not targetPlayer or not isAlive(targetPlayer.Character) then
        autoFire = false
        status.Text = "Target unavailable/dead - Auto stopped"
        status.TextColor3 = Color3.fromRGB(255, 90, 90)
        return
    end

    if not isAlive(LocalPlayer.Character) then
        autoFire = false
        status.Text = "You died - Auto stopped"
        status.TextColor3 = Color3.fromRGB(255, 90, 90)
        return
    end

    local targetChar = targetPlayer.Character
    local targetHRP = getHRP(targetChar)
    local targetPart = getTargetPart(targetChar)
    local localHRP = getHRP(LocalPlayer.Character)

    if not targetHRP or not targetPart or not localHRP then
        return
    end

    local profile = updateProfile(targetPlayer, targetHRP)
    local pingMs = getPingMs()
    local origin = localHRP.Position
    local distance = (targetHRP.Position - origin).Magnitude

    if distance > MAX_DISTANCE then
        status.Text = string.format("Target %.0f studs away (max %d)", distance, MAX_DISTANCE)
        status.TextColor3 = Color3.fromRGB(255, 190, 90)
        return
    end

    local localVelocity = localHRP.AssemblyLinearVelocity

    local aimPos, predictedHRP, horizon, travelTime, networkLead, serverOrigin =
        calculateAim(origin, localVelocity, targetHRP, targetPart, profile, pingMs)

    lastFire = now

    local shotDirection = aimPos - serverOrigin
    if shotDirection.Magnitude > 0 then
        lastShotProbe = {
            origin = serverOrigin,
            direction = shotDirection.Unit,
            expires = os.clock() + 0.35,
            calibrated = false,
        }
    end

    local ok, err = pcall(function()
        remote:FireServer(aimPos, "NewFireball")
    end)

    if not ok then
        warn("[AdaptiveFireball] FireServer failed:", err)
        return
    end

    scheduleResidualLearning(
        targetPlayer,
        profile,
        predictedHRP,
        horizon,
        profile.fastVel,
        profile.jukeEMA
    )

    status.TextColor3 = Color3.fromRGB(120, 255, 120)
    status.Text = string.format(
        "Firing %s -> %s | learned shots: %d\nCooldown: 1.50s",
        targetPlayer.Name,
        BODY_PARTS[currentBodyPartIndex].name,
        profile.shotsLearned
    )

    print(string.format(
        "[AdaptiveFireball] %s | dist %.1f | target %.1f stud/s | ping %.0fms | juke %.2f | horizon %.3fs | travel %.3fs | net %.3fs | model %.1f",
        targetPlayer.Name,
        distance,
        profile.fastVel.Magnitude,
        pingMs,
        profile.jukeEMA,
        horizon,
        travelTime,
        networkLead,
        projectileSpeedModel
    ))
end

-- ===== Auto loop =====
local function startAuto()
    if STATE.AutoConn then return end

    STATE.AutoConn = RunService.Heartbeat:Connect(function()
        if not autoFire or targetName == "" then return end

        local targetPlayer = findPlayerByPartial(targetName)
        if not targetPlayer or not isAlive(targetPlayer.Character) then
            autoFire = false
            status.Text = "Target unavailable/dead - Auto stopped"
            status.TextColor3 = Color3.fromRGB(255, 90, 90)
            return
        end

        if not isAlive(LocalPlayer.Character) then
            autoFire = false
            status.Text = "You died - Auto stopped"
            status.TextColor3 = Color3.fromRGB(255, 90, 90)
            return
        end

        local targetHRP = getHRP(targetPlayer.Character)
        local localHRP = getHRP(LocalPlayer.Character)

        if targetHRP and localHRP then
            local profile = updateProfile(targetPlayer, targetHRP)
            local pingMs = getPingMs()
            local dist = (localHRP.Position - targetHRP.Position).Magnitude

            -- Estimate the same lead telemetry even between shots.
            local targetPart = getTargetPart(targetPlayer.Character)
            local horizon = 0
            local networkLead = 0
            if targetPart then
                local _, _, h, _, nl = calculateAim(
                    localHRP.Position,
                    localHRP.AssemblyLinearVelocity,
                    targetHRP,
                    targetPart,
                    profile,
                    pingMs
                )
                horizon = h
                networkLead = nl
            end

            telemetry.Text = string.format(
                "Distance: %.1f | Target speed: %.1f | Ping: %.0fms\nJuke: %.2f | Lead: %.3fs (net %.3fs) | Projectile model: %.1f",
                dist,
                profile.fastVel.Magnitude,
                pingMs,
                profile.jukeEMA,
                horizon,
                networkLead,
                projectileSpeedModel
            )
        end

        fireAtTarget()
    end)
end

local function stopAuto()
    if STATE.AutoConn then
        STATE.AutoConn:Disconnect()
        STATE.AutoConn = nil
    end
    telemetry.Text = "Distance: — | Target speed: — | Ping: —\nJuke: — | Lead: — | Projectile model: —"
end

-- ===== Mobile controls + desktop hotkeys =====
local function lockTargetFromBox()
    if input.Text == "" then
        status.TextColor3 = Color3.fromRGB(255, 190, 90)
        status.Text = "Type a target username/display name first."
        return
    end

    local found = findPlayerByPartial(input.Text)
    if found then
        targetName = found.Name
        input.Text = found.Name
        local p = getProfile(found)
        status.TextColor3 = Color3.fromRGB(120, 255, 120)
        status.Text = string.format(
            "Target locked: %s | learned shots: %d",
            found.Name,
            p.shotsLearned
        )
    else
        status.TextColor3 = Color3.fromRGB(255, 110, 110)
        status.Text = "No player matched that name."
    end
end

local function toggleAuto()
    if targetName == "" then
        status.TextColor3 = Color3.fromRGB(255, 190, 90)
        status.Text = "Set and LOCK a target first."
        autoFire = false
        updateAutoButton()
        return
    end

    autoFire = not autoFire

    if autoFire then
        startAuto()
        status.TextColor3 = Color3.fromRGB(120, 255, 120)
        status.Text = "AUTO ENABLED - " .. targetName .. "\nCooldown: 1.50s"
    else
        stopAuto()
        status.TextColor3 = Color3.fromRGB(255, 190, 90)
        status.Text = "AUTO DISABLED"
    end

    updateAutoButton()
end

local function cycleBodyPart()
    currentBodyPartIndex = (currentBodyPartIndex % #BODY_PARTS) + 1
    updatePartLabel()
    status.TextColor3 = Color3.fromRGB(120, 255, 120)
    status.Text = "Aiming at: " .. BODY_PARTS[currentBodyPartIndex].name
end

local destroyed = false
local function destroyScript()
    if destroyed then return end
    destroyed = true

    autoFire = false
    stopAuto()

    if STATE.InputConn then
        STATE.InputConn:Disconnect()
        STATE.InputConn = nil
    end

    if STATE.ProjectileConn then
        STATE.ProjectileConn:Disconnect()
        STATE.ProjectileConn = nil
    end

    if STATE.DragConn then
        STATE.DragConn:Disconnect()
        STATE.DragConn = nil
    end

    if gui then
        gui:Destroy()
    end

    _G.FireballAdaptiveMobileV2 = nil
    print("[AdaptiveFireball-Mobile] Destroyed.")
end

-- Touch controls.
miniToggle.Activated:Connect(function()
    frame.Visible = not frame.Visible
end)

closeX.Activated:Connect(function()
    frame.Visible = false
end)

hideButton.Activated:Connect(function()
    frame.Visible = false
end)

lockButton.Activated:Connect(lockTargetFromBox)

input.FocusLost:Connect(function(enterPressed)
    if enterPressed then
        lockTargetFromBox()
    end
end)

autoButton.Activated:Connect(toggleAuto)
bodyButton.Activated:Connect(cycleBodyPart)
destroyButton.Activated:Connect(destroyScript)

-- Desktop hotkeys remain available, so this same file also works on PC.
local keyLock = {
    f4 = false,
    v = false,
    n = false,
    f10 = false,
}

STATE.InputConn = UserInputService.InputBegan:Connect(function(inp, processed)
    if processed then return end

    local kc = inp.KeyCode

    if kc == Enum.KeyCode.F4 and not keyLock.f4 then
        keyLock.f4 = true
        frame.Visible = not frame.Visible
        task.delay(0.20, function() keyLock.f4 = false end)
        return
    end

    if kc == Enum.KeyCode.F10 and not keyLock.f10 then
        keyLock.f10 = true
        destroyScript()
        return
    end

    if kc == Enum.KeyCode.Return and frame.Visible then
        lockTargetFromBox()
        return
    end

    if kc == Enum.KeyCode.V and not keyLock.v then
        keyLock.v = true
        toggleAuto()
        task.delay(0.18, function() keyLock.v = false end)
        return
    end

    if kc == Enum.KeyCode.N and not keyLock.n then
        keyLock.n = true
        cycleBodyPart()
        task.delay(0.18, function() keyLock.n = false end)
        return
    end
end)

-- Keep auto mode alive across our own respawn.
LocalPlayer.CharacterAdded:Connect(function()
    task.wait(1)
    if autoFire and targetName ~= "" then
        startAuto()
    end
end)

Players.PlayerRemoving:Connect(function(player)
    profiles[player.UserId] = nil
    if player.Name == targetName then
        autoFire = false
        stopAuto()
    end
end)

print("[AdaptiveFireball-Mobile] v2 loaded.")
print("[AdaptiveFireball-Mobile] Touch UI + per-player movement learning + correct RTT ping + 1.50s fire interval active.")
print(string.format("[AdaptiveFireball-Mobile] Projectile speed model starts at %.1f and self-calibrates when detectable.", projectileSpeedModel))
