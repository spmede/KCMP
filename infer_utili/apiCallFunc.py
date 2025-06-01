
import time
import base64
from io import BytesIO
from openai import OpenAI, OpenAIError, APITimeoutError, InternalServerError, AuthenticationError, RateLimitError

def apiCall(prompt, num_generation=1, temp=0., model="gpt-3.5-turbo"):

    client = OpenAI()  # you need to configure 'client' to successfully call API service
    
    def exponential_backoff(attempt):
        time.sleep(min(2 ** attempt, 60)) 

    for attempt in range(5):  # Maximum 5 attempts
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                n=num_generation,
                temperature=temp
            )

            # safety check in case completion=None
            if completion is None or not hasattr(completion, "choices") or not completion.choices:
                # print(f"[Warning] Empty or invalid response from model on attempt {attempt}.")
                exponential_backoff(attempt)
                continue

            # Extract the results
            ans = [completion.choices[i].message.content for i in range(num_generation)]
            return ans

        except OpenAIError as e:
            # Handle APITimeoutError
            if isinstance(e, APITimeoutError):  
                time.sleep(2 ** attempt)
            
            elif isinstance(e, RateLimitError):
                exponential_backoff(attempt)

            # Handle InternalServerError
            elif isinstance(e, InternalServerError):
                time.sleep(2 ** attempt)

            elif isinstance(e, AuthenticationError):
                time.sleep(2 ** attempt)

            # Handle other errors like RateLimitError or ServerError based on http_status
            elif hasattr(e, 'http_status') and e.http_status in [429, 500]:  # Check for status codes
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                exponential_backoff(attempt) # Re-raise the exception for other errors

    # raise Exception("Max retries exceeded. Unable to complete the request.")
    return 'noanswer'


def encode_image_from_pil(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG") 
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def apiCall_img(image, text, temperature=0.3, num_gen_token=8, model='gpt-4o-mini'):

    client = OpenAI()  # you need to configure 'client' to successfully call API service
     
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image_from_pil(image)}"}
                },
                {"type": "text", "text": text}
                        ]
        }]


    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=num_gen_token
    )
     
    return response.choices[0].message.content.lower()