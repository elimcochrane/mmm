import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image  # For custom shapes
import numpy as np  # For image processing

# 1. Load your word frequency data
df = pd.read_excel('freq_2017-2018.xlsx') 

# 2. Convert to dictionary (word: frequency)
word_freq = dict(zip(df['word'], df['frequency']))

# 3. Create basic word cloud
wc = WordCloud(
    width=1200,
    height=800,
    background_color='white',
    max_words=200,  # Limit number of words
    colormap='viridis'  # Color scheme
).generate_from_frequencies(word_freq)

# 4. Display the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()

# 5. Save to file
output_file = 'wordcloud.png'
wc.to_file(output_file)
print(f"✅ Word cloud saved as {output_file}")

# (Optional) Show interactive preview
plt.show()