import cv2
import mediapipe as mp
import math
import numpy as np

# ================================================================= #
# >>> 1. THE DUAL-FORCE ANCHOR RAGDOLL ENGINE <<<
# ================================================================= #

class Point:
    """A point in the physics simulation."""
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.old_pos = self.pos.copy()
        self.gravity = np.array([0.0, 0.5])

    def update(self, friction=0.98):
        velocity = self.pos - self.old_pos
        self.old_pos = self.pos.copy()
        self.pos += velocity * friction + self.gravity
    
    def constrain(self, width, height, bounce=0.6):
        velocity = self.pos - self.old_pos
        if self.pos[0] > width: self.pos[0]=width; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        elif self.pos[0] < 0: self.pos[0]=0; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        if self.pos[1] > height: self.pos[1]=height; self.old_pos[1]=self.pos[1]+velocity[1]*bounce
        elif self.pos[1] < 0: self.pos[1]=0; self.old_pos[1]=self.pos[1]+velocity[1]*bounce

    def apply_force(self, target_pos, strength):
        """Pulls the point towards a target. This is the core of the new control system."""
        force = target_pos - self.pos
        self.pos += force * strength

class Stick:
    """A rigid connection between two points."""
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1
        self.length = math.hypot(self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1])
    
    def update(self, stiffness=0.5):
        dx, dy = self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1]
        dist = math.hypot(dx, dy); diff = self.length - dist
        percent = diff / (dist+1e-6) / 2 * stiffness
        offset = np.array([dx*percent, dy*percent])
        self.p0.pos -= offset; self.p1.pos += offset

class Ragdoll:
    def __init__(self, x, y):
        self.points = [ Point(x,y), Point(x,y+50), Point(x,y+100),            # 0-2: Head, Chest, Hips
                        Point(x-25,y+50), Point(x-50,y+100), Point(x-75,y+150),  # 3-5: L Shoulder, Elbow, Hand
                        Point(x+25,y+50), Point(x+50,y+100), Point(x+75,y+150),  # 6-8: R Shoulder, Elbow, Hand
                        Point(x-15,y+100), Point(x-30,y+150), Point(x-45,y+200),# 9-11: L Hip, Knee, Foot
                        Point(x+15,y+100), Point(x+30,y+150), Point(x+45,y+200) ]# 12-14: R Hip, Knee, Foot
        self.sticks = [ Stick(self.points[0], self.points[1]), Stick(self.points[1], self.points[2]),
                        Stick(self.points[1], self.points[3]), Stick(self.points[1], self.points[6]),
                        Stick(self.points[3], self.points[4]), Stick(self.points[4], self.points[5]),
                        Stick(self.points[6], self.points[7]), Stick(self.points[7], self.points[8]),
                        Stick(self.points[2], self.points[9]), Stick(self.points[2], self.points[12]),
                        Stick(self.points[9], self.points[10]), Stick(self.points[10], self.points[11]),
                        Stick(self.points[12], self.points[13]), Stick(self.points[13], self.points[14])]

    def update_simulation(self, width, height):
        for p in self.points: p.update()
        for _ in range(5):
            for s in self.sticks: s.update()
        for p in self.points: p.constrain(width, height)

    def draw(self, canvas):
        for s in self.sticks: cv2.line(canvas,tuple(s.p0.pos.astype(int)),tuple(s.p1.pos.astype(int)),(255,255,255),3)
        cv2.circle(canvas, tuple(self.points[0].pos.astype(int)), 12, (255,255,255), -1)

# ================================================================= #
# >>> 2. MAIN APPLICATION LOGIC <<<
# ================================================================= #

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
w, h = int(cap.get(3)), int(cap.get(4))

mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
hands_detector = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=2)

ragdoll = Ragdoll(w / 2, h / 3)
control_strength = 0.2
control_points_for_drawing = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    frame = cv2.flip(frame, 1)
    
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    control_points_for_drawing.clear()
    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h_lm: h_lm.landmark[0].x)
        
        # --- NEW DUAL-FORCE ANCHOR CONTROLS ---
        hand_midpoint = None
        
        for i, hand_lm in enumerate(sorted_hands):
            lm = hand_lm.landmark
            
            # Get thumb and index finger positions
            thumb_pos = np.array([lm[4].x*w, lm[4].y*h])
            index_pos = np.array([lm[8].x*w, lm[8].y*h])
            wrist_pos = np.array([lm[0].x*w, lm[0].y*h])

            # Apply forces based on which hand it is (left or right on screen)
            if i == 0: # Left Hand
                ragdoll.points[5].apply_force(thumb_pos, control_strength) # Left Hand
                ragdoll.points[11].apply_force(index_pos, control_strength)# Left Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[5]), (index_pos, ragdoll.points[11])])
            else: # Right Hand
                ragdoll.points[8].apply_force(thumb_pos, control_strength) # Right Hand
                ragdoll.points[14].apply_force(index_pos, control_strength)# Right Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[8]), (index_pos, ragdoll.points[14])])

            if hand_midpoint is None: hand_midpoint = wrist_pos
            else: hand_midpoint = (hand_midpoint + wrist_pos) / 2
            
        # Body Positioning ("The Stretch")
        if hand_midpoint is not None:
             ragdoll.points[1].apply_force(hand_midpoint, control_strength * 0.5) # Chest
             control_points_for_drawing.append((hand_midpoint, ragdoll.points[1]))


    ragdoll.update_simulation(w, h)

    # --- Drawing ---
    stage = cv2.addWeighted(frame, 0.4, np.full(frame.shape, (20,20,20), np.uint8), 0.6, 0)
    
    for hand_pos, ragdoll_point in control_points_for_drawing:
        cv2.line(stage, tuple(hand_pos.astype(int)), tuple(ragdoll_point.pos.astype(int)), (0, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(stage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    ragdoll.draw(stage)
    
    cv2.imshow('Dual-Force Anchor Ragdoll', stage)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
hands_detector.close()import cv2
import mediapipe as mp
import math
import numpy as np

# ================================================================= #
# >>> 1. THE DUAL-FORCE ANCHOR RAGDOLL ENGINE <<<
# ================================================================= #

class Point:
    """A point in the physics simulation."""
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.old_pos = self.pos.copy()
        self.gravity = np.array([0.0, 0.5])

    def update(self, friction=0.98):
        velocity = self.pos - self.old_pos
        self.old_pos = self.pos.copy()
        self.pos += velocity * friction + self.gravity
    
    def constrain(self, width, height, bounce=0.6):
        velocity = self.pos - self.old_pos
        if self.pos[0] > width: self.pos[0]=width; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        elif self.pos[0] < 0: self.pos[0]=0; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        if self.pos[1] > height: self.pos[1]=height; self.old_pos[1]=self.pos[1]+velocity[1]*bounce
        elif self.pos[1] < 0: self.pos[1]=0; self.old_pos[1]=self.pos[1]+velocity[1]*bounce

    def apply_force(self, target_pos, strength):
        """Pulls the point towards a target. This is the core of the new control system."""
        force = target_pos - self.pos
        self.pos += force * strength

class Stick:
    """A rigid connection between two points."""
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1
        self.length = math.hypot(self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1])
    
    def update(self, stiffness=0.5):
        dx, dy = self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1]
        dist = math.hypot(dx, dy); diff = self.length - dist
        percent = diff / (dist+1e-6) / 2 * stiffness
        offset = np.array([dx*percent, dy*percent])
        self.p0.pos -= offset; self.p1.pos += offset

class Ragdoll:
    def __init__(self, x, y):
        self.points = [ Point(x,y), Point(x,y+50), Point(x,y+100),            # 0-2: Head, Chest, Hips
                        Point(x-25,y+50), Point(x-50,y+100), Point(x-75,y+150),  # 3-5: L Shoulder, Elbow, Hand
                        Point(x+25,y+50), Point(x+50,y+100), Point(x+75,y+150),  # 6-8: R Shoulder, Elbow, Hand
                        Point(x-15,y+100), Point(x-30,y+150), Point(x-45,y+200),# 9-11: L Hip, Knee, Foot
                        Point(x+15,y+100), Point(x+30,y+150), Point(x+45,y+200) ]# 12-14: R Hip, Knee, Foot
        self.sticks = [ Stick(self.points[0], self.points[1]), Stick(self.points[1], self.points[2]),
                        Stick(self.points[1], self.points[3]), Stick(self.points[1], self.points[6]),
                        Stick(self.points[3], self.points[4]), Stick(self.points[4], self.points[5]),
                        Stick(self.points[6], self.points[7]), Stick(self.points[7], self.points[8]),
                        Stick(self.points[2], self.points[9]), Stick(self.points[2], self.points[12]),
                        Stick(self.points[9], self.points[10]), Stick(self.points[10], self.points[11]),
                        Stick(self.points[12], self.points[13]), Stick(self.points[13], self.points[14])]

    def update_simulation(self, width, height):
        for p in self.points: p.update()
        for _ in range(5):
            for s in self.sticks: s.update()
        for p in self.points: p.constrain(width, height)

    def draw(self, canvas):
        for s in self.sticks: cv2.line(canvas,tuple(s.p0.pos.astype(int)),tuple(s.p1.pos.astype(int)),(255,255,255),3)
        cv2.circle(canvas, tuple(self.points[0].pos.astype(int)), 12, (255,255,255), -1)

# ================================================================= #
# >>> 2. MAIN APPLICATION LOGIC <<<
# ================================================================= #

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
w, h = int(cap.get(3)), int(cap.get(4))

mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
hands_detector = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=2)

ragdoll = Ragdoll(w / 2, h / 3)
control_strength = 0.2
control_points_for_drawing = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    frame = cv2.flip(frame, 1)
    
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    control_points_for_drawing.clear()
    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h_lm: h_lm.landmark[0].x)
        
        # --- NEW DUAL-FORCE ANCHOR CONTROLS ---
        hand_midpoint = None
        
        for i, hand_lm in enumerate(sorted_hands):
            lm = hand_lm.landmark
            
            # Get thumb and index finger positions
            thumb_pos = np.array([lm[4].x*w, lm[4].y*h])
            index_pos = np.array([lm[8].x*w, lm[8].y*h])
            wrist_pos = np.array([lm[0].x*w, lm[0].y*h])

            # Apply forces based on which hand it is (left or right on screen)
            if i == 0: # Left Hand
                ragdoll.points[5].apply_force(thumb_pos, control_strength) # Left Hand
                ragdoll.points[11].apply_force(index_pos, control_strength)# Left Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[5]), (index_pos, ragdoll.points[11])])
            else: # Right Hand
                ragdoll.points[8].apply_force(thumb_pos, control_strength) # Right Hand
                ragdoll.points[14].apply_force(index_pos, control_strength)# Right Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[8]), (index_pos, ragdoll.points[14])])

            if hand_midpoint is None: hand_midpoint = wrist_pos
            else: hand_midpoint = (hand_midpoint + wrist_pos) / 2
            
        # Body Positioning ("The Stretch")
        if hand_midpoint is not None:
             ragdoll.points[1].apply_force(hand_midpoint, control_strength * 0.5) # Chest
             control_points_for_drawing.append((hand_midpoint, ragdoll.points[1]))


    ragdoll.update_simulation(w, h)

    # --- Drawing ---
    stage = cv2.addWeighted(frame, 0.4, np.full(frame.shape, (20,20,20), np.uint8), 0.6, 0)
    
    for hand_pos, ragdoll_point in control_points_for_drawing:
        cv2.line(stage, tuple(hand_pos.astype(int)), tuple(ragdoll_point.pos.astype(int)), (0, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(stage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    ragdoll.draw(stage)
    
    cv2.imshow('Dual-Force Anchor Ragdoll', stage)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
hands_detector.close()import cv2
import mediapipe as mp
import math
import numpy as np

# ================================================================= #
# >>> 1. THE DUAL-FORCE ANCHOR RAGDOLL ENGINE <<<
# ================================================================= #

class Point:
    """A point in the physics simulation."""
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.old_pos = self.pos.copy()
        self.gravity = np.array([0.0, 0.5])

    def update(self, friction=0.98):
        velocity = self.pos - self.old_pos
        self.old_pos = self.pos.copy()
        self.pos += velocity * friction + self.gravity
    
    def constrain(self, width, height, bounce=0.6):
        velocity = self.pos - self.old_pos
        if self.pos[0] > width: self.pos[0]=width; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        elif self.pos[0] < 0: self.pos[0]=0; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        if self.pos[1] > height: self.pos[1]=height; self.old_pos[1]=self.pos[1]+velocity[1]*bounce
        elif self.pos[1] < 0: self.pos[1]=0; self.old_pos[1]=self.pos[1]+velocity[1]*bounce

    def apply_force(self, target_pos, strength):
        """Pulls the point towards a target. This is the core of the new control system."""
        force = target_pos - self.pos
        self.pos += force * strength

class Stick:
    """A rigid connection between two points."""
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1
        self.length = math.hypot(self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1])
    
    def update(self, stiffness=0.5):
        dx, dy = self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1]
        dist = math.hypot(dx, dy); diff = self.length - dist
        percent = diff / (dist+1e-6) / 2 * stiffness
        offset = np.array([dx*percent, dy*percent])
        self.p0.pos -= offset; self.p1.pos += offset

class Ragdoll:
    def __init__(self, x, y):
        self.points = [ Point(x,y), Point(x,y+50), Point(x,y+100),            # 0-2: Head, Chest, Hips
                        Point(x-25,y+50), Point(x-50,y+100), Point(x-75,y+150),  # 3-5: L Shoulder, Elbow, Hand
                        Point(x+25,y+50), Point(x+50,y+100), Point(x+75,y+150),  # 6-8: R Shoulder, Elbow, Hand
                        Point(x-15,y+100), Point(x-30,y+150), Point(x-45,y+200),# 9-11: L Hip, Knee, Foot
                        Point(x+15,y+100), Point(x+30,y+150), Point(x+45,y+200) ]# 12-14: R Hip, Knee, Foot
        self.sticks = [ Stick(self.points[0], self.points[1]), Stick(self.points[1], self.points[2]),
                        Stick(self.points[1], self.points[3]), Stick(self.points[1], self.points[6]),
                        Stick(self.points[3], self.points[4]), Stick(self.points[4], self.points[5]),
                        Stick(self.points[6], self.points[7]), Stick(self.points[7], self.points[8]),
                        Stick(self.points[2], self.points[9]), Stick(self.points[2], self.points[12]),
                        Stick(self.points[9], self.points[10]), Stick(self.points[10], self.points[11]),
                        Stick(self.points[12], self.points[13]), Stick(self.points[13], self.points[14])]

    def update_simulation(self, width, height):
        for p in self.points: p.update()
        for _ in range(5):
            for s in self.sticks: s.update()
        for p in self.points: p.constrain(width, height)

    def draw(self, canvas):
        for s in self.sticks: cv2.line(canvas,tuple(s.p0.pos.astype(int)),tuple(s.p1.pos.astype(int)),(255,255,255),3)
        cv2.circle(canvas, tuple(self.points[0].pos.astype(int)), 12, (255,255,255), -1)

# ================================================================= #
# >>> 2. MAIN APPLICATION LOGIC <<<
# ================================================================= #

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
w, h = int(cap.get(3)), int(cap.get(4))

mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
hands_detector = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=2)

ragdoll = Ragdoll(w / 2, h / 3)
control_strength = 0.2
control_points_for_drawing = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    frame = cv2.flip(frame, 1)
    
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    control_points_for_drawing.clear()
    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h_lm: h_lm.landmark[0].x)
        
        # --- NEW DUAL-FORCE ANCHOR CONTROLS ---
        hand_midpoint = None
        
        for i, hand_lm in enumerate(sorted_hands):
            lm = hand_lm.landmark
            
            # Get thumb and index finger positions
            thumb_pos = np.array([lm[4].x*w, lm[4].y*h])
            index_pos = np.array([lm[8].x*w, lm[8].y*h])
            wrist_pos = np.array([lm[0].x*w, lm[0].y*h])

            # Apply forces based on which hand it is (left or right on screen)
            if i == 0: # Left Hand
                ragdoll.points[5].apply_force(thumb_pos, control_strength) # Left Hand
                ragdoll.points[11].apply_force(index_pos, control_strength)# Left Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[5]), (index_pos, ragdoll.points[11])])
            else: # Right Hand
                ragdoll.points[8].apply_force(thumb_pos, control_strength) # Right Hand
                ragdoll.points[14].apply_force(index_pos, control_strength)# Right Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[8]), (index_pos, ragdoll.points[14])])

            if hand_midpoint is None: hand_midpoint = wrist_pos
            else: hand_midpoint = (hand_midpoint + wrist_pos) / 2
            
        # Body Positioning ("The Stretch")
        if hand_midpoint is not None:
             ragdoll.points[1].apply_force(hand_midpoint, control_strength * 0.5) # Chest
             control_points_for_drawing.append((hand_midpoint, ragdoll.points[1]))


    ragdoll.update_simulation(w, h)

    # --- Drawing ---
    stage = cv2.addWeighted(frame, 0.4, np.full(frame.shape, (20,20,20), np.uint8), 0.6, 0)
    
    for hand_pos, ragdoll_point in control_points_for_drawing:
        cv2.line(stage, tuple(hand_pos.astype(int)), tuple(ragdoll_point.pos.astype(int)), (0, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(stage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    ragdoll.draw(stage)
    
    cv2.imshow('Dual-Force Anchor Ragdoll', stage)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
hands_detector.close()import cv2
import mediapipe as mp
import math
import numpy as np

# ================================================================= #
# >>> 1. THE DUAL-FORCE ANCHOR RAGDOLL ENGINE <<<
# ================================================================= #

class Point:
    """A point in the physics simulation."""
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.old_pos = self.pos.copy()
        self.gravity = np.array([0.0, 0.5])

    def update(self, friction=0.98):
        velocity = self.pos - self.old_pos
        self.old_pos = self.pos.copy()
        self.pos += velocity * friction + self.gravity
    
    def constrain(self, width, height, bounce=0.6):
        velocity = self.pos - self.old_pos
        if self.pos[0] > width: self.pos[0]=width; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        elif self.pos[0] < 0: self.pos[0]=0; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        if self.pos[1] > height: self.pos[1]=height; self.old_pos[1]=self.pos[1]+velocity[1]*bounce
        elif self.pos[1] < 0: self.pos[1]=0; self.old_pos[1]=self.pos[1]+velocity[1]*bounce

    def apply_force(self, target_pos, strength):
        """Pulls the point towards a target. This is the core of the new control system."""
        force = target_pos - self.pos
        self.pos += force * strength

class Stick:
    """A rigid connection between two points."""
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1
        self.length = math.hypot(self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1])
    
    def update(self, stiffness=0.5):
        dx, dy = self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1]
        dist = math.hypot(dx, dy); diff = self.length - dist
        percent = diff / (dist+1e-6) / 2 * stiffness
        offset = np.array([dx*percent, dy*percent])
        self.p0.pos -= offset; self.p1.pos += offset

class Ragdoll:
    def __init__(self, x, y):
        self.points = [ Point(x,y), Point(x,y+50), Point(x,y+100),            # 0-2: Head, Chest, Hips
                        Point(x-25,y+50), Point(x-50,y+100), Point(x-75,y+150),  # 3-5: L Shoulder, Elbow, Hand
                        Point(x+25,y+50), Point(x+50,y+100), Point(x+75,y+150),  # 6-8: R Shoulder, Elbow, Hand
                        Point(x-15,y+100), Point(x-30,y+150), Point(x-45,y+200),# 9-11: L Hip, Knee, Foot
                        Point(x+15,y+100), Point(x+30,y+150), Point(x+45,y+200) ]# 12-14: R Hip, Knee, Foot
        self.sticks = [ Stick(self.points[0], self.points[1]), Stick(self.points[1], self.points[2]),
                        Stick(self.points[1], self.points[3]), Stick(self.points[1], self.points[6]),
                        Stick(self.points[3], self.points[4]), Stick(self.points[4], self.points[5]),
                        Stick(self.points[6], self.points[7]), Stick(self.points[7], self.points[8]),
                        Stick(self.points[2], self.points[9]), Stick(self.points[2], self.points[12]),
                        Stick(self.points[9], self.points[10]), Stick(self.points[10], self.points[11]),
                        Stick(self.points[12], self.points[13]), Stick(self.points[13], self.points[14])]

    def update_simulation(self, width, height):
        for p in self.points: p.update()
        for _ in range(5):
            for s in self.sticks: s.update()
        for p in self.points: p.constrain(width, height)

    def draw(self, canvas):
        for s in self.sticks: cv2.line(canvas,tuple(s.p0.pos.astype(int)),tuple(s.p1.pos.astype(int)),(255,255,255),3)
        cv2.circle(canvas, tuple(self.points[0].pos.astype(int)), 12, (255,255,255), -1)

# ================================================================= #
# >>> 2. MAIN APPLICATION LOGIC <<<
# ================================================================= #

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
w, h = int(cap.get(3)), int(cap.get(4))

mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
hands_detector = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=2)

ragdoll = Ragdoll(w / 2, h / 3)
control_strength = 0.2
control_points_for_drawing = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    frame = cv2.flip(frame, 1)
    
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    control_points_for_drawing.clear()
    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h_lm: h_lm.landmark[0].x)
        
        # --- NEW DUAL-FORCE ANCHOR CONTROLS ---
        hand_midpoint = None
        
        for i, hand_lm in enumerate(sorted_hands):
            lm = hand_lm.landmark
            
            # Get thumb and index finger positions
            thumb_pos = np.array([lm[4].x*w, lm[4].y*h])
            index_pos = np.array([lm[8].x*w, lm[8].y*h])
            wrist_pos = np.array([lm[0].x*w, lm[0].y*h])

            # Apply forces based on which hand it is (left or right on screen)
            if i == 0: # Left Hand
                ragdoll.points[5].apply_force(thumb_pos, control_strength) # Left Hand
                ragdoll.points[11].apply_force(index_pos, control_strength)# Left Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[5]), (index_pos, ragdoll.points[11])])
            else: # Right Hand
                ragdoll.points[8].apply_force(thumb_pos, control_strength) # Right Hand
                ragdoll.points[14].apply_force(index_pos, control_strength)# Right Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[8]), (index_pos, ragdoll.points[14])])

            if hand_midpoint is None: hand_midpoint = wrist_pos
            else: hand_midpoint = (hand_midpoint + wrist_pos) / 2
            
        # Body Positioning ("The Stretch")
        if hand_midpoint is not None:
             ragdoll.points[1].apply_force(hand_midpoint, control_strength * 0.5) # Chest
             control_points_for_drawing.append((hand_midpoint, ragdoll.points[1]))


    ragdoll.update_simulation(w, h)

    # --- Drawing ---
    stage = cv2.addWeighted(frame, 0.4, np.full(frame.shape, (20,20,20), np.uint8), 0.6, 0)
    
    for hand_pos, ragdoll_point in control_points_for_drawing:
        cv2.line(stage, tuple(hand_pos.astype(int)), tuple(ragdoll_point.pos.astype(int)), (0, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(stage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    ragdoll.draw(stage)
    
    cv2.imshow('Dual-Force Anchor Ragdoll', stage)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
hands_detector.close()import cv2
import mediapipe as mp
import math
import numpy as np

# ================================================================= #
# >>> 1. THE DUAL-FORCE ANCHOR RAGDOLL ENGINE <<<
# ================================================================= #

class Point:
    """A point in the physics simulation."""
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.old_pos = self.pos.copy()
        self.gravity = np.array([0.0, 0.5])

    def update(self, friction=0.98):
        velocity = self.pos - self.old_pos
        self.old_pos = self.pos.copy()
        self.pos += velocity * friction + self.gravity
    
    def constrain(self, width, height, bounce=0.6):
        velocity = self.pos - self.old_pos
        if self.pos[0] > width: self.pos[0]=width; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        elif self.pos[0] < 0: self.pos[0]=0; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        if self.pos[1] > height: self.pos[1]=height; self.old_pos[1]=self.pos[1]+velocity[1]*bounce
        elif self.pos[1] < 0: self.pos[1]=0; self.old_pos[1]=self.pos[1]+velocity[1]*bounce

    def apply_force(self, target_pos, strength):
        """Pulls the point towards a target. This is the core of the new control system."""
        force = target_pos - self.pos
        self.pos += force * strength

class Stick:
    """A rigid connection between two points."""
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1
        self.length = math.hypot(self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1])
    
    def update(self, stiffness=0.5):
        dx, dy = self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1]
        dist = math.hypot(dx, dy); diff = self.length - dist
        percent = diff / (dist+1e-6) / 2 * stiffness
        offset = np.array([dx*percent, dy*percent])
        self.p0.pos -= offset; self.p1.pos += offset

class Ragdoll:
    def __init__(self, x, y):
        self.points = [ Point(x,y), Point(x,y+50), Point(x,y+100),            # 0-2: Head, Chest, Hips
                        Point(x-25,y+50), Point(x-50,y+100), Point(x-75,y+150),  # 3-5: L Shoulder, Elbow, Hand
                        Point(x+25,y+50), Point(x+50,y+100), Point(x+75,y+150),  # 6-8: R Shoulder, Elbow, Hand
                        Point(x-15,y+100), Point(x-30,y+150), Point(x-45,y+200),# 9-11: L Hip, Knee, Foot
                        Point(x+15,y+100), Point(x+30,y+150), Point(x+45,y+200) ]# 12-14: R Hip, Knee, Foot
        self.sticks = [ Stick(self.points[0], self.points[1]), Stick(self.points[1], self.points[2]),
                        Stick(self.points[1], self.points[3]), Stick(self.points[1], self.points[6]),
                        Stick(self.points[3], self.points[4]), Stick(self.points[4], self.points[5]),
                        Stick(self.points[6], self.points[7]), Stick(self.points[7], self.points[8]),
                        Stick(self.points[2], self.points[9]), Stick(self.points[2], self.points[12]),
                        Stick(self.points[9], self.points[10]), Stick(self.points[10], self.points[11]),
                        Stick(self.points[12], self.points[13]), Stick(self.points[13], self.points[14])]

    def update_simulation(self, width, height):
        for p in self.points: p.update()
        for _ in range(5):
            for s in self.sticks: s.update()
        for p in self.points: p.constrain(width, height)

    def draw(self, canvas):
        for s in self.sticks: cv2.line(canvas,tuple(s.p0.pos.astype(int)),tuple(s.p1.pos.astype(int)),(255,255,255),3)
        cv2.circle(canvas, tuple(self.points[0].pos.astype(int)), 12, (255,255,255), -1)

# ================================================================= #
# >>> 2. MAIN APPLICATION LOGIC <<<
# ================================================================= #

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
w, h = int(cap.get(3)), int(cap.get(4))

mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
hands_detector = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=2)

ragdoll = Ragdoll(w / 2, h / 3)
control_strength = 0.2
control_points_for_drawing = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    frame = cv2.flip(frame, 1)
    
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    control_points_for_drawing.clear()
    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h_lm: h_lm.landmark[0].x)
        
        # --- NEW DUAL-FORCE ANCHOR CONTROLS ---
        hand_midpoint = None
        
        for i, hand_lm in enumerate(sorted_hands):
            lm = hand_lm.landmark
            
            # Get thumb and index finger positions
            thumb_pos = np.array([lm[4].x*w, lm[4].y*h])
            index_pos = np.array([lm[8].x*w, lm[8].y*h])
            wrist_pos = np.array([lm[0].x*w, lm[0].y*h])

            # Apply forces based on which hand it is (left or right on screen)
            if i == 0: # Left Hand
                ragdoll.points[5].apply_force(thumb_pos, control_strength) # Left Hand
                ragdoll.points[11].apply_force(index_pos, control_strength)# Left Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[5]), (index_pos, ragdoll.points[11])])
            else: # Right Hand
                ragdoll.points[8].apply_force(thumb_pos, control_strength) # Right Hand
                ragdoll.points[14].apply_force(index_pos, control_strength)# Right Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[8]), (index_pos, ragdoll.points[14])])

            if hand_midpoint is None: hand_midpoint = wrist_pos
            else: hand_midpoint = (hand_midpoint + wrist_pos) / 2
            
        # Body Positioning ("The Stretch")
        if hand_midpoint is not None:
             ragdoll.points[1].apply_force(hand_midpoint, control_strength * 0.5) # Chest
             control_points_for_drawing.append((hand_midpoint, ragdoll.points[1]))


    ragdoll.update_simulation(w, h)

    # --- Drawing ---
    stage = cv2.addWeighted(frame, 0.4, np.full(frame.shape, (20,20,20), np.uint8), 0.6, 0)
    
    for hand_pos, ragdoll_point in control_points_for_drawing:
        cv2.line(stage, tuple(hand_pos.astype(int)), tuple(ragdoll_point.pos.astype(int)), (0, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(stage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    ragdoll.draw(stage)
    
    cv2.imshow('Dual-Force Anchor Ragdoll', stage)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
hands_detector.close()import cv2
import mediapipe as mp
import math
import numpy as np

# ================================================================= #
# >>> 1. THE DUAL-FORCE ANCHOR RAGDOLL ENGINE <<<
# ================================================================= #

class Point:
    """A point in the physics simulation."""
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.old_pos = self.pos.copy()
        self.gravity = np.array([0.0, 0.5])

    def update(self, friction=0.98):
        velocity = self.pos - self.old_pos
        self.old_pos = self.pos.copy()
        self.pos += velocity * friction + self.gravity
    
    def constrain(self, width, height, bounce=0.6):
        velocity = self.pos - self.old_pos
        if self.pos[0] > width: self.pos[0]=width; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        elif self.pos[0] < 0: self.pos[0]=0; self.old_pos[0]=self.pos[0]+velocity[0]*bounce
        if self.pos[1] > height: self.pos[1]=height; self.old_pos[1]=self.pos[1]+velocity[1]*bounce
        elif self.pos[1] < 0: self.pos[1]=0; self.old_pos[1]=self.pos[1]+velocity[1]*bounce

    def apply_force(self, target_pos, strength):
        """Pulls the point towards a target. This is the core of the new control system."""
        force = target_pos - self.pos
        self.pos += force * strength

class Stick:
    """A rigid connection between two points."""
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1
        self.length = math.hypot(self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1])
    
    def update(self, stiffness=0.5):
        dx, dy = self.p1.pos[0]-self.p0.pos[0], self.p1.pos[1]-self.p0.pos[1]
        dist = math.hypot(dx, dy); diff = self.length - dist
        percent = diff / (dist+1e-6) / 2 * stiffness
        offset = np.array([dx*percent, dy*percent])
        self.p0.pos -= offset; self.p1.pos += offset

class Ragdoll:
    def __init__(self, x, y):
        self.points = [ Point(x,y), Point(x,y+50), Point(x,y+100),            # 0-2: Head, Chest, Hips
                        Point(x-25,y+50), Point(x-50,y+100), Point(x-75,y+150),  # 3-5: L Shoulder, Elbow, Hand
                        Point(x+25,y+50), Point(x+50,y+100), Point(x+75,y+150),  # 6-8: R Shoulder, Elbow, Hand
                        Point(x-15,y+100), Point(x-30,y+150), Point(x-45,y+200),# 9-11: L Hip, Knee, Foot
                        Point(x+15,y+100), Point(x+30,y+150), Point(x+45,y+200) ]# 12-14: R Hip, Knee, Foot
        self.sticks = [ Stick(self.points[0], self.points[1]), Stick(self.points[1], self.points[2]),
                        Stick(self.points[1], self.points[3]), Stick(self.points[1], self.points[6]),
                        Stick(self.points[3], self.points[4]), Stick(self.points[4], self.points[5]),
                        Stick(self.points[6], self.points[7]), Stick(self.points[7], self.points[8]),
                        Stick(self.points[2], self.points[9]), Stick(self.points[2], self.points[12]),
                        Stick(self.points[9], self.points[10]), Stick(self.points[10], self.points[11]),
                        Stick(self.points[12], self.points[13]), Stick(self.points[13], self.points[14])]

    def update_simulation(self, width, height):
        for p in self.points: p.update()
        for _ in range(5):
            for s in self.sticks: s.update()
        for p in self.points: p.constrain(width, height)

    def draw(self, canvas):
        for s in self.sticks: cv2.line(canvas,tuple(s.p0.pos.astype(int)),tuple(s.p1.pos.astype(int)),(255,255,255),3)
        cv2.circle(canvas, tuple(self.points[0].pos.astype(int)), 12, (255,255,255), -1)

# ================================================================= #
# >>> 2. MAIN APPLICATION LOGIC <<<
# ================================================================= #

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
w, h = int(cap.get(3)), int(cap.get(4))

mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
hands_detector = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=2)

ragdoll = Ragdoll(w / 2, h / 3)
control_strength = 0.2
control_points_for_drawing = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    frame = cv2.flip(frame, 1)
    
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    control_points_for_drawing.clear()
    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h_lm: h_lm.landmark[0].x)
        
        # --- NEW DUAL-FORCE ANCHOR CONTROLS ---
        hand_midpoint = None
        
        for i, hand_lm in enumerate(sorted_hands):
            lm = hand_lm.landmark
            
            # Get thumb and index finger positions
            thumb_pos = np.array([lm[4].x*w, lm[4].y*h])
            index_pos = np.array([lm[8].x*w, lm[8].y*h])
            wrist_pos = np.array([lm[0].x*w, lm[0].y*h])

            # Apply forces based on which hand it is (left or right on screen)
            if i == 0: # Left Hand
                ragdoll.points[5].apply_force(thumb_pos, control_strength) # Left Hand
                ragdoll.points[11].apply_force(index_pos, control_strength)# Left Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[5]), (index_pos, ragdoll.points[11])])
            else: # Right Hand
                ragdoll.points[8].apply_force(thumb_pos, control_strength) # Right Hand
                ragdoll.points[14].apply_force(index_pos, control_strength)# Right Foot
                control_points_for_drawing.extend([(thumb_pos, ragdoll.points[8]), (index_pos, ragdoll.points[14])])

            if hand_midpoint is None: hand_midpoint = wrist_pos
            else: hand_midpoint = (hand_midpoint + wrist_pos) / 2
            
        # Body Positioning ("The Stretch")
        if hand_midpoint is not None:
             ragdoll.points[1].apply_force(hand_midpoint, control_strength * 0.5) # Chest
             control_points_for_drawing.append((hand_midpoint, ragdoll.points[1]))


    ragdoll.update_simulation(w, h)

    # --- Drawing ---
    stage = cv2.addWeighted(frame, 0.4, np.full(frame.shape, (20,20,20), np.uint8), 0.6, 0)
    
    for hand_pos, ragdoll_point in control_points_for_drawing:
        cv2.line(stage, tuple(hand_pos.astype(int)), tuple(ragdoll_point.pos.astype(int)), (0, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(stage, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    ragdoll.draw(stage)
    
    cv2.imshow('Dual-Force Anchor Ragdoll', stage)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
hands_detector.close()