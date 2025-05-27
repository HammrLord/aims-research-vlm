from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import json
import base64
from dotenv import load_dotenv
import time



def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def load_all_examples(json_path, img_dir):
    with open(json_path, "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        img_path = os.path.join(img_dir, item["image"])
        image_b64 = encode_image_base64(img_path)
        examples.append({
            "title": item["image"],
            "summary": item["recipe_summary"],
            "image_b64": image_b64
        })
    return examples

def generate_summary(test_image_path, test_title, json_path, img_dir):
    examples = load_all_examples(json_path, img_dir)
    test_image_b64 = encode_image_base64(test_image_path)

    content = [
        {
            "type": "text",
            "text": (
                """You are a chef's assistant that summarizes recipes in 3 lines or fewer.Below are some examples with\
                    a dish noisy title, an image, and a short recipe. Analyze the examples and then write a similar summary\
                    for the test image."""
            )
        }
    ]
    for index, ex in enumerate(examples):
        content.append({"type": "text", "text": f"\nExample {index+1} Title: {ex['title']}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex['image_b64']}"}})
        content.append({"type": "text", "text": f"Summary: {ex['summary']}\n"})

    content.append({"type": "text", "text": f"\nTest Title: {test_title}"})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{test_image_b64}"}})
    content.append({"type": "text", "text": "Summary:"})

    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY"))
    response = llm.invoke([HumanMessage(content=content)])

    print("\nGenerated Recipe Summary:")
    return response

if __name__ == "__main__":
    with open("test/test.json", "r") as f:
        test_data = json.load(f)
    image_path = os.path.join("test",test_data[4]["image"])
    logs = generate_summary(
            test_image_path= image_path,
            test_title=test_data[4]["image"],
            json_path="train/train.json",
            img_dir="train"
        )
    with open("log.txt","a") as file:
        file.write(f"Image: {test_data[4]['image']}\n")
        file.write(f"Summary: {logs}\n\n")
    # counter = 0
    # with open("test/test.json", "r") as f:
    #     test_data = json.load(f)
    # for i in test_data:
    #     print(f"⚙Testing Image: {counter}\n")
    #     image_path = os.path.join("test",i["image"])
    #     generate_summary(
    #         test_image_path= image_path,
    #         test_title=i["image"],
    #         json_path="train/train.json",
    #         img_dir="train"
    #     )
    #     counter += 1
    #     time.sleep(4)
