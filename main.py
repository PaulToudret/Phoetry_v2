#Install required packages
# pip install -r requirements.txt

#Import libraries
from src.image_recognition import image_label_detector, choose_label
from src.poem_generator import poem_generator
from transformers import GPT2Tokenizer,GPT2LMHeadModel

#Initialize model



# Define paths
project_folder_path="./"
images_path=project_folder_path+"/images"
model_path=project_folder_path+"/trained_model/poet-gpt2"

model_name='gpt2'
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

tokenizer=GPT2Tokenizer.from_pretrained(model_path)
model=GPT2LMHeadModel.from_pretrained(model_path)

# Proposed labels
labels=["tree","flower","sunset","sunrise","cloud","mountain","beach","river","lake","waterfall","forest","grassland","desert","rain","snow",
                   "road","traffic jam","hill","valley","cave","farm","garden","coastline","field","pond","sky","animal","insect","fungi",
                   "leaf","pebble","stone", "dog", "cat","bird","butterfly","bee", "stars","moon","sun"]

# Poem generation from picture
image_path=images_path+"/sunset.jpg"
def generate_poem_from_picture(image_path,labels):
    top3_predicted_labels=image_label_detector(image_path,labels)
    theme=choose_label(top3_predicted_labels)
    poem=poem_generator(model_path=model_path, theme=theme, 
          max_length=200,  temperature=0.5,  top_k=60, top_p=0.9, repetition_penalty=1.5)
    return poem
print(generate_poem_from_picture(image_path,labels))
