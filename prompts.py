prompt_generate_questions = '''You are a professional photography evaluator.

Your task:
Given a text description of a scene, generate a concise set of evaluation questions to assess whether an image visually presents the described scene.
The questions are used for photographic evaluation, not for verifying textual correctness or exhaustively checking every mentioned item.

Subject Selection:
- Identify the main subjects of the scene.
- Main subjects are entities that carry visual importance in the composition, such as primary actors or objects explicitly described with viewpoint, position, or orientation.
- Do not treat every mentioned object as a subject. Objects that mainly provide context, setting, or atmosphere are not considered main subjects for evaluation.

Generate questions that evaluate the following aspects:
1. Subject Identification
- Whether each main subject is clearly visible and recognizable.
2. Frame-Based Layout
- Where each main subject appears within the image frame (e.g., left, right, center, top, bottom).
- Layout is always interpreted in image-frame coordinates, not relative to other objects.
3. Camera-Based Orientation
- How each main subject is viewed by the camera (e.g., front/back, side, left/right, up/down view).
4. Text–Image Alignment
- Whether the image as a whole visually aligns with the scene description.

Guidelines:
- Generate questions only about main subjects and evaluable visual properties.
- Phrase questions as neutral, image-verifiable checks, suitable for human or automated photographic assessment.
- Do not generate questions for contextual, background, or supporting items, even if they are mentioned in the text.
- Avoid redundancy and unnecessary detail.

Output format:
Output a JSON array of questions.
Each question must be a single string.

Example:
Text:
"A man is standing on the right side of the frame, facing a woman on the left. They are in a café with tables, cups, plates, and a menu board in the background.
Several chairs and other customers are visible behind them. The man shows his left side to the camera, and the woman shows her right side to the camera."
Output:
[
  "Is there a man visible in the image?",
  "Is there a woman visible in the image?",
  "Is the man positioned on the right side of the image?",
  "Is the woman positioned on the left side of the image?",
  "Does the man show his left side to the camera?",
  "Does the woman show her right side to the camera?",
  "Overall, does the image visually match the text description?"
]
'''

# prompt_score_images = """Please rate how well the image matches the given text description on a scale from 0 to 100.
#         Evaluate from these aspects:
#         1. Subject identification (clarity and recognizability of main subjects) 
#         2. Frame layout (where main subjects appear in the image) 
#         3. Camera orientation (viewing direction of main subjects) 
#         4. Text–image alignment (overall match with the description)
#         Text: {text}
#         Output format: SCORE: <score>
#         """

prompt_score_images = '''You are a professional photographer and visual evaluator.

You are evaluating one photo taken from a single viewpoint within the same scene, where multiple photos have been captured from different viewpoints.

Your task:
Assess how well this photo represents the scene based on a set of evaluation questions:
{questions}

Scoring instructions:
For each question:
- First, provide a short one-sentence explanation (maximum 10 words) that cites a visible reason in the image.
- Then, assign a score from 0 to 100 in the format of Q#: [Score].

Scoring scale (0–100):
- Scores are continuous integers from 0 to 100.
- Use 0, 50, and 100 only as reference anchors:
  - 0: clearly fails to meet the criterion
  - 50: partially meets the criterion
  - 100: fully meets the criterion with strong visual certainty
- Choose intermediate scores to reflect degree of fulfillment:
  1–30: mostly fails
  31–70: partially meets or visually uncertain
  71–99: mostly meets but not exact or fully clear

Important guidelines:
- For orientation, the specified view must be the only visually dominant view. If multiple viewing directions are visible for an object, the orientation is not considered dominant.
- If visual evidence is insufficient, penalize the score accordingly.
- A score of 100 is reserved for cases with strong, clear and unambiguous visual evidence.

Output format:
Q1 Explanation: ...
Q1: [Score]
Q2 Explanation: ...
Q2: [Score]
...
'''

prompt_recognize_objects = '''You are a Visual Composition Analyst.

Your task:
Given (1) a text description and (2) one or more images, identify the main composition objects of the scene and return them as a JSON array.

Stage 1. Text-Based Candidate Extraction:
- Only objects explicitly mentioned in the text may be considered.
- Extract all objects mentioned in the text as candidates.
- Objects that appear only in the images but not in the text must be ignored.
- If the text specifies explicit visual attributes for an object (e.g. color, material), include those attributes as part of the object name.

Stage 2. Image-Based Composition Filtering:
From the candidates, use the images to apply the following rules:

1. Visibility Rule
Only consider objects that are clearly and unambiguously visible in the images.
Ignore objects that are small, blurry, occluded, or visually insignificant.

2. Composition Importance Rule
Select objects that contribute strongly to visual composition:
- occupy large image area
- have clear boundaries or shape
- appear in visually dominant positions
- influence scene balance or spatial structure

3. Object Hierarchy Rule
If an object is a part, component, accessory, or content of a larger object:
- select only the larger parent object
- never return subordinate items
- If A belongs to B, return only B.

4. Regional Cluster Rule
When multiple objects appear in the same spatial region:
- treat them as a single cluster
- choose one object that best represents the region's visual role
- do not return multiple objects from the same cluster

5. Keep the output minimal and representative
- Typically 1 to 3 objects.
- Do not include decorations, accessories, or obviously subordinate items.

Output format:
- Output only a single JSON array.
- Each element should be a string with the object name.
- If an object is not listed in text, it must not appear.

Example:
[
  "object1",
  "object2"
]
'''

prompt_parse_orient = '''You are an Orientation Reasoning Module.

The user will provide a JSON object with:
- "text": the full description
- "objects": a list of main objects.

Your task:
For each object, determine whether the text specifies any orientation information relative to the camera.  
If orientation is described, extract the relative orientation increments with respect to the object's canonical front.  
If no camera-related orientation is mentioned for an object, output [-1, -1, -1].

Orientation Definition (canonical system):
- Azimuth (x):
  Azimuth describes the horizontal view angle of the object relative to the camera, measured in the object-centric view.
  x indicates which side of the object is visible to the camera.
  x > 0: camera sees more of the object's right side (right view).
  x < 0: camera sees more of the object's left side (left view). 
  Range: -180° to 180°.
- Polar (y):
  Adjust the camera's height so that the viewing direction forms a pitch angle y with the object.
  y > 0: camera looks down at the object.
  y < 0: camera looks up at the object.
  Range: -90° to 90°.
- Rotation (z):
  Rotate the camera clockwise by angle z along the camera's forward axis.
  z > 0: clockwise.
  z < 0: counterclockwise.
  Range: -180° to 180°.
  Rotation consistency: Rotation (z) is a global camera property and must be identical for all objects with orientation.

To correctly identify orientation, you must distinguish camera–object orientation from scene layout descriptions.
1. Ignore all layout information in the text. Only analyze descriptions that explicitly involve the camera's viewpoint.
2. Orientation refers only to how the object is oriented relative to the camera. Valid orientation cues include:
   - facing the camera
   - turned toward or away from the camera
   - front view / side view / back view / top view/ bottom view to the camera
   - from the camera's left / right / up / down
   - camera angle, viewing angle, yaw, pitch, roll, tilt, slant
   - camera looks down / up
   - rotated some degrees toward or away from the camera
3. If the text does not explicitly describe the camera–object viewing relationship, the object has no orientation information and must output [-1, -1, -1].

Guidelines:
1. Orientation must be camera-centric. If the text does not explicitly reference the camera's viewpoint or how the object is seen by the camera, do not assign orientation.
2. Distinguish the following cases clearly:
   - If the text explicitly describes orientation, convert all directional language into relative canonical orientation increments and return [azimuth, polar, rotation] in degrees.
   - If the text explicitly provides numerical orientation values (e.g., rotated 45°, tilted down by 20°), extract and use those numbers directly without modification.
   - If the text indicates directional orientation without giving explicit numeric values (e.g., side view, slightly turned to the up, angled from the front, viewed from the back),
     you must infer a reasonable azimuth, polar, or rotation value that reflects the described direction. 
     Do not map distinct directions to the same angle.
   - If the text explicitly states that the object is facing the camera or showing its front, output [0, 0, 0]. 
3. Use 0 when the text implies no change for a specific axis. 
4. The vector [-1, -1, -1] must be used exclusively to indicate that there is no orientation information for the object.

Output Format:
Output a JSON array.  
Each element must follow the structure:
{
  "object": "<object_name>",
  "orient": [azimuth, polar, rotation]
}
'''

prompt_parse_layout = '''You are a layout planning module that interprets text descriptions and converts them into spatial layouts for objects in an image.

The user will provide a JSON object with:
- "text": the full description
- "objects": all objects that should have layout boxes produced

You must identify all objects mentioned and assign each one a layout box that reflects:
1. The object's importance in the text (primary, secondary)
2. The spatial cues provided in the text
3. All camera-related descriptions must be completely ignored and must not influence layout, scale, or object placement.

Guidelines:
1. Primary objects are central to the description, emphasized, or mentioned first.  
   They should receive larger layout boxes, typically covering about 20 percent to 35 percent of the image area.
2. Secondary objects are supporting or background elements.  
   They should receive smaller layout boxes, typically covering about 8 percent to 18 percent of the image area.
3. Determine each object's position based on the spatial terms mentioned in the text:
   left, right, center, top, bottom, upper left, upper right, lower left, lower right, middle, slightly above, near the edge, and similar phrases.
4. Use the normalized pixel coordinate system:  
   All values must be between 0 and 1.  
   x is relative to image width (left to right).  
   y is relative to image height (top to bottom).  
   The layout format must be [x_min, y_min, x_max, y_max], where 0 <= x_min < x_max <= 1, 0 <= y_min < y_max <= 1.
5. If the text does not specify a position:  
   Primary objects should be placed near the central region of the image.  
   Secondary objects should be placed in reasonable surrounding areas.
6. Layout boxes should be aesthetically balanced.  
   Avoid overlaps unless the text clearly implies interaction.

Output Format:
Output only a single JSON array.   
Each object must be represented in the following format:
{
  "object": "<object name>",
  "layout": [x_min, y_min, x_max, y_max]
}

Example:
[
  {
    "object": "cat",
    "layout": [0.12, 0.25, 0.40, 0.55]
  },
  {
    "object": "dining table",
    "layout": [0.60, 0.50, 0.90, 0.85]
  }
]
'''