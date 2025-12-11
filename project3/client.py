import requests
import sys

def predict_image(url: str, endpoint_url: str):
    """
    Downloads an image from a URL and sends it to the classification endpoint.
    """
    try:
        print(f"Downloading image from {url}...")
        img_response = requests.get(url)
        img_response.raise_for_status()
        image_data = img_response.content
        
        print(f"Sending request to {endpoint_url}...")
        
        files = {
            "img": ("test_image.jpg", image_data, "image/jpeg")
        }
        
        response = requests.post(endpoint_url, files=files) 

        response.raise_for_status()
        
        print("Prediction Result:")
        print(response.json())
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    TEST_IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    
    if len(sys.argv) > 1:
        ENDPOINT = sys.argv[1]
    else:
        ENDPOINT = "http://localhost:3000/classify"
        
    print(f"Testing endpoint: {ENDPOINT}")
    predict_image(TEST_IMAGE_URL, ENDPOINT)
