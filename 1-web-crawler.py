# importing the library
from google_images_download import google_images_download 

# Definindo os sites de confiança
SITES_LIST = ["http://www.dermatlas.net/", 
              "http://www.dermis.net/dermisroot/en/home/index.htm",
              "http://www.meddean.luc.edu/lumen/MedEd/medicine/ \
	      dermatology/melton/atlas.htm",
              "http://www.dermatoweb.net/", 
              "http://www.atlasdermatologico.com.br/",
              "http://www.hellenicdermatlas.com/en/?params=en"]

# Palavras-chaves para pesquisar
SKIN_LESION = ["Actinic Keratosis",
         "Basal cell carcinoma",
         "Dermatofibroma",
         "Hemangioma",
         "Intraepithelial carcinoma",
         "Bowen’s disease",
         "Lentigo",
         "Malignant melanoma",
         "Melanocytic nevus",
         "Pyogenic granuloma",
         "Seborrheic keratosis",
         "Squamous cell carcinoma",
         "Wart"]

for skin_lesion in SKIN_LESION:
    for site in SITE:
        print(site)
        response = google_images_download.googleimagesdownload() 

        # creating list of arguments
        arguments = {
            "keywords": str(skin_lesion),
            "limiti": 1000,
            "specific_site": str(site),
            "output_directory": "dataset_2",
            "print_urls":True
        }

        # passing the arguments to the function
        paths = response.download(arguments)  
        print(paths)
