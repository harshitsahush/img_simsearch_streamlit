from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import os
import chromadb
import uuid

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
model = tf.keras.models.load_model("test.keras")

def process_images(images):
    #takes a set of images
    #for each image:
    #   saves it locally
    #   creates embeddings
    #   adds to list
    #store in chromadb
   
    img_paths = []      #list of dicts
    img_embeds = []

    for image in images:
        curr_path = img_to_local(image)
        img_paths.append({"path" : curr_path})

        embeds = create_embeds(curr_path)
        img_embeds.append(list(embeds))

    img_embeds = np.array(img_embeds).astype(np.float32)

    print(img_paths)
    
    client = chromadb.PersistentClient(path = os.getcwd())
    collection = client.get_or_create_collection("image_embeddings")
    collection.add(
        metadatas = img_paths,
        embeddings= img_embeds,
        ids = [str(i) for i in range(len(img_paths))]
    )


def create_embeds(img_path):
    #pass image path to model and return embed
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(img)
    img = np.expand_dims(img, axis = 0)


    embed = model.predict(img)
    return embed[0]


def img_to_local(image):
    #saves image to local with name "id" and returns path
    id = uuid.uuid4()
    img = Image.open(image)
    save_path = "./temp_data/" + str(id) + ".png"
    img.save(save_path)
    return save_path


def process_target_image(image):
    #takes single image
    #creates embeddings
    #finds similar images from chromadb
    curr_path = img_to_local(image)
    embeds = create_embeds(curr_path)
    embeds = [embeds]
    embeds = np.array(embeds).astype("float")

    client = chromadb.PersistentClient(path = os.getcwd())
    collection = client.get_or_create_collection("image_embeddings")
    sim_docs = collection.query(
        query_embeddings= embeds,
        n_results = 5
    )

    return sim_docs
