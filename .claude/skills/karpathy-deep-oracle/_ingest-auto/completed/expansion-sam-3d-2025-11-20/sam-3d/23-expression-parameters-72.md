# 72 Expression Parameters for Facial Animation

**MHR facial rig: 72 blendshape coefficients for expressive face animation**

---

## 1. Facial Expression Overview

**72 Expression Parameters:**

Control facial features:
- **Jaw**: Open/close mouth (1 param)
- **Mouth**: Smile, frown, pucker, corners (15 params)
- **Eyes**: Blink, gaze direction, squint (10 params)
- **Brows**: Raise, furrow, arch (8 params)
- **Cheeks**: Puff, suck, raise (5 params)
- **Nose**: Wrinkle, flare (3 params)
- **Other**: 30 params for fine details

---

## 2. Blendshape Model

**Linear Combination:**

Facial Expression: E = Σᵢ(αᵢ × Bᵢ)
- αᵢ: Expression weight [0-1]
- Bᵢ: Blendshape basis (pre-defined expression)

**Example:**
- α_smile = 0.8 (80% smile)
- α_brow_raise = 0.5 (50% brow raise)
- Result: Smiling face with slightly raised eyebrows

---

## 3. Facial Action Units (FACS)

**Psychology-Inspired:**

Based on Facial Action Coding System:
- AU1: Inner brow raiser
- AU2: Outer brow raiser
- AU4: Brow lowerer
- AU12: Lip corner puller (smile)

MHR's 72 params map to FACS units.

---

## 4. Speech Animation

**Visemes (Mouth Shapes):**

Subset of 72 params for speech:
- A: Mouth open ("ah")
- E: Mouth wide ("ee")
- O: Lips rounded ("oh")
- M/B/P: Lips closed

**Lip Sync:**
- Input: Audio waveform
- Output: 72 expression params per frame
- Result: Talking avatar

---

## 5. ARR-COC-0-1 Integration (10%)

**Emotion & Social Cues:**

Facial expressions convey:
- Emotion (happy, sad, angry)
- Intent (surprise, interest)
- Communication (speech, nods)

Relevance realization includes social signals.

---

**Sources:**
- MHR facial rig
- FACS (Facial Action Coding System)
- Blendshape animation
