from utils import *


def main():
    st.title("Similar image search")

    with st.sidebar:
        images = st.file_uploader("Upload the dataset images", accept_multiple_files=True, type = ['jpg', 'jpeg', 'png'])
        submit = st.button("Upload images")

        if(submit):
            if(images is None):
                st.error("Please upload atleast ONE image")
            else:
                process_images(images)      #will create and store embeddings
    
    target_img = st.file_uploader("Upload the target image", accept_multiple_files=False, type = ['jpg', 'jpeg', 'png'])
    submit_img = st.button("Find the most similar images!")

    if(submit_img):
        if(target_img is None):
            st.error("Please upload an image!")
        else:
            sim_docs = process_target_image(target_img)        #returns urls of similar images
            print(sim_docs)

            for ele in sim_docs["metadatas"][0]:
                st.image(ele["path"])

if(__name__ == "__main__"):
    main()  