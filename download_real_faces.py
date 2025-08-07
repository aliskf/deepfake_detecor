import requests
import os
import time

output_dir = 'image_classification_data/real'
os.makedirs(output_dir, exist_ok=True)

num_images = 1000

print(f"Downloading {num_images} real-looking faces from randomuser.me...")

for i in range(num_images):
    try:
        response = requests.get('https://randomuser.me/api/?inc=picture')
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        image_url = data['results'][0]['picture']['large']
        
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        filename = os.path.join(output_dir, f'real_{i+1}.jpg')

        with open(filename, 'wb') as f:
            f.write(image_response.content)

        print(f'Successfully downloaded {filename}')

    except requests.exceptions.RequestException as e:
        print(f'Error downloading image {i+1}: {e}')

    time.sleep(0.1) # Be polite to the server

print('Finished downloading real-looking images.')
