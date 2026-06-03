import os
import sys
import json
import PIL.Image
from google import genai

# 1. Setup Modern API Client
API_KEY = "AIzaSyA9hDpFHzea24vwYSBpXjDFIbnvKffv34U"  # PASTE YOUR KEY HERE
client = genai.Client(api_key=API_KEY)

# 2. Load the Image
try:
    print("Loading image from disk...")
    image_path = os.path.expanduser("~/ros2_ws/src/oak_object_detection/oak_object_detection/vlm_test_frame.jpg")
    img = PIL.Image.open(image_path)
except FileNotFoundError:
    print(f"Error: Could not find image at {image_path}")
    sys.exit()

# 3. The Strict Prompt
prompt = """
You are the executive perception node for an autonomous mobile robot.
Look at the provided camera frame. Identify the clearest, safest path forward to avoid any obstacles.
Calculate a safe target waypoint. 
X is the distance forward in meters (maximum 2.0). 
Y is the distance left or right in meters (Left is positive, Right is negative, maximum 1.0).

You MUST respond ONLY with a raw JSON object. Do not include markdown formatting, backticks, or conversational text.
Example format:
{"x": 1.5, "y": 0.0}
"""

# 4. Call the VLM (Using the modern 2.5-flash model)
print("Sending frame to Cloud VLM...")
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt, img]
    )
except Exception as e:
    print(f"API Error: {e}")
    sys.exit()

# 5. Parse the Output
print("\n--- RAW VLM RESPONSE ---")
print(response.text)
print("------------------------\n")

try:
    # Strip any accidental markdown formatting
    clean_text = response.text.strip().replace("```json", "").replace("```", "")
    coord = json.loads(clean_text)
    
    print(f"SUCCESS! Parsed Coordinates:")
    print(f"Forward (X): {coord['x']} meters")
    print(f"Lateral (Y): {coord['y']} meters")
    print("\nNext Step: We will feed these numbers directly into Nav2.")
    
except json.JSONDecodeError:
    print("FAILED: The VLM did not return a valid JSON string.")
