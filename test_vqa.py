import os

from huggingface_hub import InferenceClient


MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def main():

    token = os.getenv("HF_TOKEN")

    if not token:
        print("ERROR: HF_TOKEN is not set.")
        return

    client = InferenceClient(
        api_key=token
    )

    print("=" * 60)
    print("SATQUERY - REMOTE VQA TEST")
    print("=" * 60)

    print("\nModel:")
    print(MODEL)

    print("\nProvider:")
    print("Hugging Face Inference Providers")

    print("\nStatus:")
    print("Sending image + question...")

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image briefly."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        answer = response.choices[0].message.content

        print("\nAI Answer:")
        print(answer)

        print("\n" + "=" * 60)
        print("REMOTE VQA TEST SUCCESSFUL")
        print("=" * 60)

    except Exception as e:

        print("\nVQA request failed.")
        print(type(e).__name__ + ":", e)


if __name__ == "__main__":
    main()