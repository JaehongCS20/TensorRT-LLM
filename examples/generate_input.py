# Creating more precise entries with exactly 128 tokens each by adjusting and refining the content.

def generate_token_entries(entries):
    # Target token count for each entry
    target_token_count = 768

    # Adjust each entry to meet the target token count
    adjusted_entries = []
    for entry in entries:
        tokens = entry.split()
        current_count = len(tokens)
        
        # If the entry is less than the target, add filler content relevant to the topic
        if current_count < target_token_count:
            while len(tokens) < target_token_count:
                tokens.extend(["This", "is", "part", "of", "Nvidia's", "commitment", "to", "innovation", "and", "excellence", "in", "the", "field", "of", "computing", "and", "technology."])
                # Trim to exact size if overflown
                tokens = tokens[:target_token_count]
        
        # If the entry is more than the target, reduce the content
        elif current_count > target_token_count:
            tokens = tokens[:target_token_count]
        
        adjusted_entries.append(' '.join(tokens))
    
    return adjusted_entries

# Revised entries to ensure each is exactly 128 tokens
# The existing entries are extended with more detailed information to reach the required token count.

extended_entries = [
    "Nvidia's groundbreaking GPU technology revolutionizes both gaming and professional visualization, leading to enhanced graphics and real-time rendering capabilities. The company's GPUs power the most visually demanding applications, making them ideal for gamers and creative professionals who require high performance and reliability.",
    "Nvidia GeForce GPUs set the gold standard in gaming by providing top-tier performance and pioneering real-time ray tracing technology for more immersive game environments. These GPUs enhance player experience through superior image quality and lighting effects, pushing the boundaries of what is possible in modern video games.",
    "Nvidia CUDA technology has democratized deep learning, giving researchers and developers powerful tools to accelerate AI and machine learning applications. This technology supports a wide range of scientific and commercial applications, enabling faster computations and more efficient data processing across various industries.",
    "With its innovative approach to autonomous driving, Nvidia drives the future of mobility, enabling safer and more efficient transportation solutions through advanced AI and data processing. Nvidia's technology supports autonomous vehicle systems, improving safety and efficiency by processing large volumes of data in real-time."
]

# Generate entries with exactly 128 tokens
adjusted_entries = generate_token_entries(extended_entries)
adjusted_token_counts = [len(entry.split()) for entry in adjusted_entries]

# Check if the adjustment was successful
# print(adjusted_token_counts, adjusted_entries)
batch = 16
mul = batch // 4
result_batch = adjusted_entries * 4
with open("../input.txt", 'w') as f:
    for i in result_batch:
        f.write(i+'\n')
