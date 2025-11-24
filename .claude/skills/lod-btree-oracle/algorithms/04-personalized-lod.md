# Personalized LOD Algorithms

**User modeling, adaptive quality, importance weighting, and perceptual optimization**

---

## Overview

Personalized Level of Detail adapts geometric and visual complexity to individual users based on their preferences, hardware capabilities, viewing patterns, and task context. Rather than applying uniform LOD thresholds to all users, personalized systems build user models that capture quality preferences, performance constraints, and attention patterns to optimize the trade-off between visual fidelity and frame rate on a per-user basis.

**Key insight**: Different users have different quality/performance preferences and perceptual sensitivities.

**Historical context**: Early LOD was one-size-fits-all → user quality settings (low/medium/high) → modern personalized systems with continuous adaptation.

---

## Primary Sources

**Core Research:**
- `source-documents/07-Exploiting Lod-Based Similarity Personalization Strategies for Recommender Systems.md` - Silva (2023), personalization strategies using feature selection
- `source-documents/02-Analysis and Design of a Personalized Recommendation System Based on a Dynamic User Interest Model - arXiv.md` - Dynamic user interest modeling

**Related Work:**
- `source-documents/13-Managing Level of Detail through Head-Tracked Peripheral Degradation_ A Model and Resulting Design Principles - arXiv.md` - Personalized VR LOD based on gaze patterns
- `source-documents/18-Visual Attention in 3D Video Games - SFU Summit.md` - Player attention patterns for adaptive rendering

**Key Principle:**
- Personalization = Feature selection + User modeling + Adaptive optimization
- Context-aware quality allocation based on individual preferences

---

## Key Concepts

### User Quality Profiles

**Definition**: Per-user preferences for visual quality vs performance.

**Profile components**:
1. **Quality preference**: Visual fidelity priority (0.0 = max FPS, 1.0 = max quality)
2. **Performance target**: Desired frame rate (30, 60, 90, 120 FPS)
3. **Hardware capability**: GPU/CPU performance tier
4. **Task context**: Gameplay, exploration, cinematic, VR

**Example profile**:
```
UserProfile {
    userID: "player_12345"
    qualityPreference: 0.7  # Prefers quality over FPS
    targetFPS: 60
    minAcceptableFPS: 45
    hardwareTier: HIGH
    taskContext: EXPLORATION
    visualSensitivity: {
        geometricDetail: 0.8  # High sensitivity to poly count
        textureDetail: 0.6    # Medium sensitivity to textures
        shadowQuality: 0.3    # Low sensitivity to shadows
    }
}
```

### Dynamic User Interest Modeling

**Adaptive learning**: Track user behavior over time to refine LOD allocation.

**Tracked metrics**:
- Time spent in different areas (indicates interest)
- Camera velocity patterns (fast = lower detail acceptable)
- Object interaction frequency (important objects get more detail)
- Performance complaints (manual quality adjustments)

**Interest decay**: Recent behavior weighted more heavily than distant past.

```
interest(object, time) = base_interest * decay_factor^(current_time - last_interaction)
```

### Feature Selection for LOD

**Challenge**: Which visual features matter most to this user?

**Feature types**:
- Geometric complexity (polygon count)
- Texture resolution
- Shadow quality
- Lighting fidelity
- Post-processing effects

**Personalized weighting**:
```
visualImpact = weighted_sum(
    geometricDetail * user.sensitivity.geometric,
    textureQuality * user.sensitivity.texture,
    shadowQuality * user.sensitivity.shadow,
    ...
)
```

### Importance-Based Allocation

**Principle**: Allocate detail proportional to object importance to user.

**Importance factors**:
1. **Semantic**: Quest objects, enemies, player character
2. **Spatial**: Near camera, in focus, screen center
3. **Temporal**: Recently interacted, moving
4. **Personal**: User-defined favorites, historical interaction

**Combined importance**:
```
importance(object, user) =
    semantic_weight * semantic_importance(object) +
    spatial_weight * spatial_importance(object, camera) +
    temporal_weight * temporal_importance(object, time) +
    personal_weight * personal_importance(object, user)
```

---

## Algorithm Details

### User Profile Initialization

**Cold start problem**: New users have no history.

**Solution**: Default profile + rapid adaptation.

```
function InitializeUserProfile(userID):
    # Start with platform-appropriate defaults
    profile = DefaultProfile(GetPlatform())

    # Quick survey (optional)
    if userOptedInSurvey:
        profile.qualityPreference = AskUserPreference()
        # "Prefer visual quality or frame rate?" (slider)

    # Hardware detection
    profile.hardwareTier = DetectGPU()
    profile.targetFPS = RecommendedFPS(profile.hardwareTier)

    # Initial sensitivity estimates (population average)
    profile.visualSensitivity = PopulationAverageSensitivity()

    return profile
```

**Rapid adaptation phase**: First 30 minutes, aggressively adjust based on observed behavior.

### Behavioral Data Collection

**Telemetry system**: Track user interactions passively.

```
function CollectBehavioralData(user, session):
    behaviorLog = []

    # Frame-by-frame logging (sampled)
    if frameCount % SAMPLE_RATE == 0:
        snapshot = {
            'timestamp': GetTime(),
            'camera': camera.position,
            'fps': GetCurrentFPS(),
            'visibleObjects': GetVisibleObjects(),
            'focusedObject': GetGazeFocus(),  # If eye-tracking available
            'inputActivity': GetInputActivity()
        }
        behaviorLog.append(snapshot)

    # Event-based logging (all events)
    OnObjectInteraction(object):
        interactionEvent = {
            'type': 'INTERACTION',
            'object': object.id,
            'duration': interaction.duration,
            'importance': object.semanticImportance
        }
        behaviorLog.append(interactionEvent)

    OnQualityComplaint():  # User manually adjusts settings
        complaintEvent = {
            'type': 'QUALITY_ADJUSTMENT',
            'previousSettings': qualitySettings,
            'newSettings': updatedSettings,
            'triggerContext': currentScene
        }
        behaviorLog.append(complaintEvent)

    return behaviorLog
```

**Privacy**: All data stored locally, user can opt-out or clear history.

### Preference Learning

**Machine learning approach**: Infer quality preferences from behavior.

```
function LearnUserPreferences(behaviorLog, currentProfile):
    # Feature extraction
    features = ExtractFeatures(behaviorLog)

    # Supervised learning from explicit adjustments
    if HasQualityAdjustments(behaviorLog):
        adjustments = GetQualityAdjustments(behaviorLog)

        for adjustment in adjustments:
            context = adjustment.triggerContext
            preferredQuality = adjustment.newSettings

            # Update preference model
            UpdatePreferenceModel(
                context,
                preferredQuality,
                currentProfile
            )

    # Unsupervised learning from implicit signals
    # High FPS variance → user sensitive to performance
    fpsVariance = CalculateFPSVariance(behaviorLog)
    if fpsVariance > threshold:
        currentProfile.qualityPreference -= 0.05  # Prioritize performance

    # Long time in areas with low LOD → user doesn't mind
    lowLODTime = TimeSpentInLowLODAreas(behaviorLog)
    if lowLODTime > 0.7 * totalTime:
        currentProfile.visualSensitivity.geometric *= 0.95  # Less sensitive

    # Frequent camera panning → prefers smooth motion
    cameraPanFrequency = CalculateCameraPanFrequency(behaviorLog)
    if cameraPanFrequency > threshold:
        currentProfile.targetFPS = max(60, currentProfile.targetFPS)

    return currentProfile
```

**Update frequency**: Every session (5-10 minutes of gameplay).

### Personalized LOD Selection

**Core algorithm**: Modify standard LOD selection with user profile.

```
function PersonalizedLODSelection(object, camera, viewport, userProfile):
    # Standard LOD calculation
    baseLOD = StandardLODSelection(object, camera, viewport)

    # Importance-based adjustment
    importance = CalculateImportance(object, userProfile, camera)

    # User sensitivity modulation
    sensitivity = userProfile.visualSensitivity.geometric

    # Quality preference bias
    qualityBias = userProfile.qualityPreference

    # Combined adjustment factor
    adjustmentFactor = importance * sensitivity * (1 + qualityBias)

    # Adjust LOD threshold
    adjustedLOD = baseLOD - floor(adjustmentFactor * 2)
    adjustedLOD = clamp(adjustedLOD, 0, object.maxLOD)

    return adjustedLOD
```

**Key modifications**:
- High-importance objects: More detail (lower LOD number)
- High user sensitivity: More detail globally
- High quality preference: Tighter thresholds

### Importance Calculation

**Semantic importance** (predefined in scene):
```
function SemanticImportance(object):
    if object.isPlayerCharacter:
        return 1.0
    else if object.isQuestTarget:
        return 0.9
    else if object.isEnemy:
        return 0.7
    else if object.isNPC:
        return 0.5
    else if object.isEnvironmentProp:
        return 0.2
    else:
        return 0.1
```

**Spatial importance** (distance + screen area):
```
function SpatialImportance(object, camera, viewport):
    distance = Distance(camera.position, object.center)
    screenArea = ProjectedScreenArea(object, camera, viewport)

    # Closer and larger → more important
    distanceFactor = 1.0 / (1.0 + distance / referenceDistance)
    areaFactor = screenArea / totalScreenArea

    return (distanceFactor + areaFactor) / 2
```

**Temporal importance** (recent interaction):
```
function TemporalImportance(object, currentTime, interactionHistory):
    lastInteraction = interactionHistory.GetLastInteraction(object)

    if lastInteraction == null:
        return 0.0

    timeSinceInteraction = currentTime - lastInteraction.time
    decayFactor = exp(-timeSinceInteraction / decayConstant)

    return decayFactor
```

**Personal importance** (user-specific):
```
function PersonalImportance(object, userProfile):
    # Check if user has marked object as favorite
    if object.id in userProfile.favoriteObjects:
        return 1.0

    # Accumulated interaction history
    interactionCount = userProfile.GetInteractionCount(object)
    totalInteractions = userProfile.GetTotalInteractions()

    if totalInteractions == 0:
        return 0.0

    # Normalized interaction frequency
    return interactionCount / totalInteractions
```

**Combined importance** (weighted sum):
```
function CalculateImportance(object, userProfile, camera):
    semantic = SemanticImportance(object)
    spatial = SpatialImportance(object, camera, viewport)
    temporal = TemporalImportance(object, currentTime, userProfile)
    personal = PersonalImportance(object, userProfile)

    # Weights can be tuned per application
    weights = {
        'semantic': 0.4,
        'spatial': 0.3,
        'temporal': 0.2,
        'personal': 0.1
    }

    importance = (
        weights.semantic * semantic +
        weights.spatial * spatial +
        weights.temporal * temporal +
        weights.personal * personal
    )

    return clamp(importance, 0.0, 1.0)
```

---

## Advanced Personalization Strategies

### Performance-Aware Adaptation

**Dynamic quality scaling**: Adjust LOD thresholds to maintain target FPS.

```
function PerformanceAwareAdaptation(userProfile):
    currentFPS = GetCurrentFPS()
    targetFPS = userProfile.targetFPS
    minFPS = userProfile.minAcceptableFPS

    # Calculate FPS deficit
    fpsDelta = targetFPS - currentFPS

    if currentFPS < minFPS:
        # Critical: aggressive quality reduction
        lodBias = 1.5  # Reduce detail 50%
    else if fpsDelta > 5:
        # Below target: modest reduction
        lodBias = 1.0 + (fpsDelta / 30.0)
    else if fpsDelta < -5:
        # Above target: can increase quality
        lodBias = 1.0 - (abs(fpsDelta) / 60.0)
    else:
        # Within acceptable range: no change
        lodBias = 1.0

    # Smooth adaptation (avoid oscillation)
    lodBias = Lerp(previousLODBias, lodBias, 0.1)

    return lodBias
```

**Application**: Multiply LOD thresholds by `lodBias`.

### Context-Aware Quality

**Different quality for different contexts**.

```
function ContextAwareQuality(userProfile, gameContext):
    if gameContext == COMBAT:
        # Prioritize performance for responsive controls
        return {
            'geometricDetail': 0.7 * userProfile.visualSensitivity.geometric,
            'targetFPS': max(60, userProfile.targetFPS),
            'importanceWeights': {'enemies': 0.9, 'environment': 0.1}
        }

    else if gameContext == EXPLORATION:
        # Prioritize visuals for immersion
        return {
            'geometricDetail': 1.2 * userProfile.visualSensitivity.geometric,
            'targetFPS': userProfile.targetFPS,
            'importanceWeights': {'environment': 0.7, 'landmarks': 0.9}
        }

    else if gameContext == CINEMATIC:
        # Maximum quality (pre-rendered or fixed camera)
        return {
            'geometricDetail': 1.5 * userProfile.visualSensitivity.geometric,
            'targetFPS': 30,  # Cinematic frame rate acceptable
            'importanceWeights': {'characters': 1.0, 'everything': 0.8}
        }

    else if gameContext == VR:
        # Mandatory high FPS, spatial personalization critical
        return {
            'geometricDetail': userProfile.visualSensitivity.geometric,
            'targetFPS': 90,  # VR minimum
            'importanceWeights': {'gaze': 1.0, 'peripheral': 0.3}
        }
```

### Personalized Popping Prevention

**Individual sensitivity to LOD transitions**.

```
function PersonalizedPoppingPrevention(userProfile):
    # Some users highly sensitive to popping
    poppingSensitivity = userProfile.visualSensitivity.lodTransitions

    if poppingSensitivity > 0.7:
        # Use aggressive geomorphing
        transitionMethod = GEOMORPH
        transitionDuration = 0.5  # seconds
        hysteresisMargin = 1.3    # 30% margin

    else if poppingSensitivity > 0.4:
        # Standard transition
        transitionMethod = GEOMORPH
        transitionDuration = 0.3
        hysteresisMargin = 1.15   # 15% margin

    else:
        # User doesn't notice/care, save performance
        transitionMethod = INSTANT
        transitionDuration = 0.0
        hysteresisMargin = 1.1    # 10% margin

    return {
        'method': transitionMethod,
        'duration': transitionDuration,
        'hysteresis': hysteresisMargin
    }
```

### Attention-Based Personalization (VR/AR)

**Per-user gaze patterns for foveated rendering**.

```
function PersonalizedFoveatedRendering(userProfile, eyeTracker):
    gazePosition = eyeTracker.GetGazePosition()

    # Personalized foveal region size
    # Users with poor vision may have larger effective fovea
    fovealRadius = userProfile.fovealSize  # Degrees from gaze center

    # Personalized peripheral degradation curve
    # Some users tolerate more aggressive degradation
    degradationCurve = userProfile.peripheralTolerance

    for object in visibleObjects:
        # Angular distance from gaze
        gazeAngle = AngleBetween(gazePosition, object)

        # Personalized LOD based on gaze distance
        if gazeAngle < fovealRadius:
            lodMultiplier = 1.0  # Full detail
        else if gazeAngle < fovealRadius * 2:
            lodMultiplier = degradationCurve.parafoveal  # User-specific
        else:
            lodMultiplier = degradationCurve.peripheral

        object.lodThreshold *= lodMultiplier
```

**Calibration**: Learn user's foveal size and degradation tolerance through experiments or implicit observation.

---

## Implementation Strategies

### Hybrid Approach

**Combine multiple signals**:

1. **Explicit**: User settings slider (quality vs performance)
2. **Implicit**: Learned from behavior (interaction patterns)
3. **Contextual**: Game state (combat, exploration, cinematic)
4. **Performance**: Dynamic adaptation (maintain target FPS)

```
function HybridPersonalizedLOD(object, camera, userProfile, gameContext):
    # Explicit user preference
    explicitBias = userProfile.qualityPreference

    # Implicit learned preference
    implicitBias = userProfile.learnedPreferences.geometricDetail

    # Contextual adjustment
    contextBias = ContextAwareQuality(userProfile, gameContext).geometricDetail

    # Performance constraint
    performanceBias = PerformanceAwareAdaptation(userProfile)

    # Combined bias (multiplicative)
    totalBias = explicitBias * implicitBias * contextBias * performanceBias

    # Standard LOD with personalized bias
    baseLOD = StandardLODSelection(object, camera, viewport)
    importance = CalculateImportance(object, userProfile, camera)

    adjustedThreshold = object.lodThreshold * totalBias / importance

    return SelectLODByThreshold(object, adjustedThreshold)
```

### Offline Profile Analysis

**Batch processing**: Analyze session logs to refine user model.

```
function OfflineProfileRefinement(userID, sessionLogs):
    # Load all historical data
    allSessions = LoadSessionLogs(userID)

    # Aggregate statistics
    stats = {
        'avgFPS': CalculateAverageFPS(allSessions),
        'fpsVariance': CalculateFPSVariance(allSessions),
        'qualityAdjustments': CountQualityAdjustments(allSessions),
        'preferredContexts': IdentifyPreferredContexts(allSessions),
        'interactionPatterns': ExtractInteractionPatterns(allSessions)
    }

    # Infer preferences
    updatedProfile = InferPreferences(stats, currentProfile)

    # Clustering: Similar users
    similarUsers = FindSimilarUsers(updatedProfile, userDatabase)

    # Transfer learning from similar users
    if len(similarUsers) > 5:
        refinedProfile = RefineWithSimilarUsers(
            updatedProfile,
            similarUsers
        )
    else:
        refinedProfile = updatedProfile

    SaveUserProfile(userID, refinedProfile)
    return refinedProfile
```

**Benefit**: More sophisticated analysis than real-time learning.

### A/B Testing for Personalization

**Evaluate effectiveness**: Compare personalized vs standard LOD.

```
function EvaluatePersonalization(userID):
    # Randomly assign to control or treatment group
    isPersonalized = (userID % 2 == 0)

    if isPersonalized:
        lodSystem = PersonalizedLODSystem(LoadUserProfile(userID))
    else:
        lodSystem = StandardLODSystem(DefaultSettings())

    # Run session
    sessionMetrics = RunGameSession(lodSystem)

    # Log metrics
    LogMetrics(userID, isPersonalized, {
        'averageFPS': sessionMetrics.avgFPS,
        'minFPS': sessionMetrics.minFPS,
        'lodTransitions': sessionMetrics.numLODTransitions,
        'userSatisfaction': sessionMetrics.postSessionSurvey,
        'playDuration': sessionMetrics.duration
    })
```

**Analysis**: Compare satisfaction scores and play duration between groups.

---

## Performance Considerations

### Computational Overhead

**Profile lookup**: O(1) hash table access per object.
**Importance calculation**: O(1) per object (precomputed history).
**Behavior logging**: ~1KB per minute (sampled at 1 Hz).

**Total overhead**: <1% frame time for 10K objects.

### Memory Footprint

**Per-user profile**: 1-5 KB
- Quality preferences: 50 bytes
- Visual sensitivities: 100 bytes
- Interaction history: 500-4000 bytes (depends on history length)
- Learned model parameters: 500 bytes

**Scalable**: 1M users = 1-5 GB total (stored on server).

### Privacy & Ethics

**Data collection concerns**:
- Behavioral tracking can feel invasive
- Gaze tracking especially sensitive

**Best practices**:
1. **Opt-in**: Users must consent to data collection
2. **Transparency**: Explain what data is collected and why
3. **Local storage**: Keep data on user's device when possible
4. **Deletion**: Allow users to clear their profile anytime
5. **Anonymization**: Never link profiles to real identities

---

## Cross-References

**Related Concepts:**
- [concepts/02-visual-perception.md](../concepts/02-visual-perception.md) - Human visual system variability
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - User-specific relevance

**Related Techniques:**
- [techniques/01-peripheral-degradation.md](../techniques/01-peripheral-degradation.md) - Gaze-aware personalization
- [techniques/00-foveated-rendering.md](../techniques/00-foveated-rendering.md) - VR personalization

**Related Algorithms:**
- [algorithms/01-lod-selection.md](01-lod-selection.md) - Base LOD selection modified by personalization
- [algorithms/00-bsp-construction.md](00-bsp-construction.md) - Spatial organization for efficient queries

**Applications:**
- Open-world games (adaptive quality for diverse players)
- VR systems (personalized foveated rendering)
- Cloud gaming (adapt to network + hardware variability)
- Accessibility (accommodate users with visual impairments)

---

## Key Takeaways

1. **One size does not fit all**: Users have different quality/performance preferences and perceptual sensitivities. Personalized LOD can improve satisfaction and engagement by respecting individual differences.

2. **Multiple signals enable robust personalization**: Combine explicit settings (quality slider), implicit behavior (interaction patterns), context (game state), and performance (FPS) for best results. No single signal sufficient.

3. **Importance-based allocation is critical**: Not all objects equally important to all users. Semantic (quest objects), spatial (near camera), temporal (recently interacted), and personal (favorites) importance all matter.

4. **Cold start problem solvable**: Default profiles + rapid adaptation (first 30 minutes) provide good initial experience. Similarity-based transfer learning from other users accelerates convergence.

5. **Privacy must be respected**: Behavioral tracking enables personalization but raises privacy concerns. Opt-in consent, local storage, and user control over data essential for ethical deployment.

6. **Performance overhead is minimal**: Profile lookup O(1), importance calculation O(1). <1% frame time for 10K objects. 1-5 KB per user profile, scalable to millions of users.

7. **Context-awareness amplifies benefit**: Different quality requirements for combat (high FPS), exploration (high visuals), cinematic (max quality), VR (mandatory 90 FPS). Context-specific profiles improve experience.

8. **Connection to relevance realization**: Personalized LOD is participatory knowing - the LOD system couples with the specific user (agent) interacting with the scene (arena). Importance calculation is perspectival knowing (user-specific salience). Learned preferences are procedural knowing (user-specific skills and sensitivities). This is deeply transjective: relevance emerges from the relationship between this particular user and this particular scene.

9. **Adaptive learning enables improvement**: Profiles refine over time as behavior is observed. Offline batch analysis enables sophisticated preference inference. A/B testing validates that personalization actually improves user experience.

10. **Accessibility implications**: Personalization can accommodate users with visual impairments (larger foveal regions, higher base quality) or motion sensitivity (aggressive popping prevention). Universal design through adaptive systems.
