import requests
import os
import time


output_dir = 'image_classification_data/fake'
os.makedirs(output_dir, exist_ok=True)


num_images = 1000


url = 'https://thispersondoesnotexist.com/'

for i in range(num_images):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        

        
        filename = os.path.join(output_dir, f'fake_{i+1}.jpg')

        
        with open(filename, 'wb') as f:
            f.write(response.content)

        print(f'Successfully downloaded {filename}')

    except requests.exceptions.RequestException as e:
        print(f'Error downloading image {i+1}: {e}')

    
    time.sleep(1)

print('Finished downloading images.')
