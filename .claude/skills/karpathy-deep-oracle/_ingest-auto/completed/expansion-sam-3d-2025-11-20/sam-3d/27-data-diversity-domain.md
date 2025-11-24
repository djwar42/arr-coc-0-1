# Data Diversity & Domain Coverage

**Geographic, demographic, and scenario diversity in SAM 3D training data**

---

## 1. Geographic Diversity

**Training Data Coverage:**
- North America: 40%
- Europe: 30%
- Asia: 20%
- Other: 10%

**Why It Matters:**
- Different clothing styles (cultural)
- Different body types (ethnic diversity)
- Different environments (urban vs rural)

**Bias Mitigation:**
- Balanced sampling across regions
- Oversampling underrepresented groups

---

## 2. Demographic Diversity

**Age Groups:**
- Children (5-12): 10%
- Teens (13-19): 15%
- Adults (20-60): 60%
- Elderly (60+): 15%

**Gender:**
- Male: 48%
- Female: 48%
- Non-binary/Other: 4%

**Body Types:**
- BMI distribution matches population statistics
- Height: 5th-95th percentile coverage
- Shape diversity via SMPL β parameters

---

## 3. Activity Diversity

**Pose Categories:**
- Standing (35%)
- Walking/Running (25%)
- Sitting (20%)
- Complex (yoga, sports, dance): 15%
- Lying/Unusual: 5%

**Activities:**
- Daily activities (cooking, typing, reading)
- Sports (basketball, yoga, cycling)
- Social interactions (handshakes, hugs)

---

## 4. Environmental Diversity

**Indoor:**
- Homes (living rooms, kitchens)
- Offices (desks, meetings)
- Gyms, studios

**Outdoor:**
- Streets, parks
- Sports fields
- Urban, rural

**Lighting:**
- Bright daylight (40%)
- Indoor lighting (30%)
- Dim/nighttime (20%)
- Mixed/harsh shadows (10%)

---

## 5. Domain Gaps

**Challenges:**
- Synthetic-to-real gap (70% synthetic data)
- Studio mocap vs wild images
- Controlled lighting vs real-world

**Bridging Strategies:**
- Domain randomization (synthetic)
- Real-world fine-tuning
- Adversarial domain adaptation

---

## 6. ARR-COC-0-1 Integration (10%)

**Diverse Data for Robust Relevance:**

Training diversity prevents bias:
- Works on all demographics (age, gender, ethnicity)
- Generalizes to novel environments
- Handles diverse activities and clothing

---

**Sources:**
- Dataset demographic statistics
- Domain adaptation research
- Bias mitigation in computer vision
