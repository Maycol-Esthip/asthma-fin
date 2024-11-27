from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image


st.set_page_config(
    page_title="Asthma Detection using Machine Learning",
    page_icon="https://images.emojiterra.com/twitter/v13.1/512px/2695.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS para el fondo degradado
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #ADD8E6, #B0E0E6, #FFFFFF);
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

with st.sidebar:



    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Diagnostic Measures", "Evaluate Data"],
        icons=["house", "droplet", "droplet-fill"],
        menu_icon="cast",
        styles={
            "container": {"padding": "5px", "background-color": "#f0f8ff"},  # Fondo del menú
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#87cefa",  # Azul claro al pasar el mouse
            },
            "nav-link-selected": {"background-color": "#4682b4", "color": "white"},  # Azul más oscuro para la selección
        },
    )

# Estilo CSS para los cuadros
box_style = """
<style>
.box {
  border: 1px solid black;
  padding: 10px;
  border-radius: 5px;
  background-color: #f9f9f9;
  margin-bottom: 20px;
  height: 400px; /* Altura fija */
  overflow-y: auto; /* Barra deslizadora */
  transition: transform 0.3s ease, box-shadow 0.3s ease; /* Transición suave */
}


.box:hover {
  transform: scale(1.07); /* Agranda el cuadro un 7% */
  box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.3); /* Sombra más pronunciada */
}
</style>
"""


# Aplicar el estilo
st.markdown(box_style, unsafe_allow_html=True)

# Contenido del primer cuadro
content1 = """
<div class="box">
    <h4 style='text-align: center;'>Asthma</h4>
    <p>Asthma is a chronic disease that affects the airways, causing them to become inflamed and narrowed, making it difficult to breathe. The main symptoms include shortness of breath, cough, wheezing and chest tightness. Common triggers are allergens, pollution, respiratory infections, exercise and strong emotions. Although there is no cure, asthma can be controlled with medication and avoiding triggers, allowing most people to lead an active, healthy life.</p>
    <img src="https://i.ibb.co/F4cVwB6/asma.webp" alt="Imagen en el cuadro 1" width="100%">
</div>
"""

# Contenido del segundo cuadro
content2 = """
<div class="box">
    <h4 style='text-align: center;'>Machine Learning</h4>
    <p>Machine learning is a branch of artificial intelligence (AI) that allows computers to learn and improve their performance on specific tasks without being explicitly programmed to do so. Instead of following predefined instructions, machine learning systems analyze data and find patterns, adjusting their responses and behaviors as they receive more information.</p>
    <img src="https://i.ibb.co/Mfz6qs9/machinelearning.jpg" alt="Imagen en el cuadro 2" width="100%">
</div>
"""

# Contenido del tercer cuadro
content3 = """
<div class="box">
    <h4 style='text-align: center;'>Benefits</h4>
    <p>Machine learning allows you to automate repetitive and difficult tasks, freeing up time and resources for more strategic tasks. It also offers better accuracy and efficiency as it can process large volumes of data quickly and with high precision. This is useful, for example, in medical diagnoses that identify diseases in early stages with great accuracy.</p>
    <img src="https://i.ibb.co/hBGKBNW/benefixios.jpg" alt="Imagen en el cuadro 3" width="100%">
</div>
"""

# Mostrar los cuadros en Streamlit
#st.markdown(content1, unsafe_allow_html=True)
#st.markdown(content2, unsafe_allow_html=True)
#st.markdown(content3, unsafe_allow_html=True)




if selected == "Home":

    st.image("https://i.ibb.co/88rrxDk/imagen-2024-11-27-012241348.png",use_container_width=True)

    #st.markdown("<h1 style='text-align: center;'><span style='color: #ff6347;'>🔥</span>Predict Asthma based on diagnostic measures<span style='color: #32cd32;'>🌱</span></h1>", unsafe_allow_html=True)
    st.markdown("""<h1 style="text-align: center;">Predict Asthma based on diagnostic measures</h1>""", unsafe_allow_html=True)
    st.write("")
    st.write("")
 
    col1, col2, col3  = st.columns([1, 1, 1])
   

    # Columna 1: Texto

   # col1.markdown("""<h4 style="text-align: center;">Asthma</h4>""", unsafe_allow_html=True)
   # col1.write("Asthma is a chronic disease that affects the airways, causing them to become inflamed and narrowed, making it difficult to breathe. The main symptoms include shortness of breath, cough, wheezing and chest tightness. Common triggers are allergens, pollution, respiratory infections, exercise and strong emotions. Although there is no cure, asthma can be controlled with medication and avoiding triggers, allowing most people to lead an active, healthy life.")
    col1.markdown(content1, unsafe_allow_html=True)

    # Columna 2: Texto

    #col2.markdown("""<h4 style="text-align: center;">Machine Learning</h4>""", unsafe_allow_html=True)
    #col2.write("Machine learning is a branch of artificial intelligence (AI) that allows computers to learn and improve their performance on specific tasks without being explicitly programmed to do so. Instead of following predefined instructions, machine learning systems analyze data and find patterns, adjusting their responses and behaviors as they receive more information.")
    col2.markdown(content2, unsafe_allow_html=True)
    
    # Columna 3: Texto

    #col3.markdown("""<h4 style="text-align: center;">Benefits</h4>""", unsafe_allow_html=True)
    #col3.write("Machine learning allows you to automate repetitive and difficult tasks, freeing up time and resources for more strategic tasks. It also offers better accuracy and efficiency as it can process large volumes of data quickly and with high precision. This is useful, for example, in medical diagnoses that identify diseases in early stages with great accuracy.")
    col3.markdown(content3, unsafe_allow_html=True)

 #col2.image("https://i.ibb.co/sPy79nG/descargar.jpg", width= 600)
    # Columna 2: Texto
    # 


    st.write("---")

            

 


    # Main title
    st.title("🌬️ Understanding Asthma")

    # Creating two columns
    colA1, colA2 = st.columns([2,1])

    # Column 1: What is Asthma? and Symptoms
    with colA1:
        st.header("What is Asthma?")
        st.write("""
        Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to breathing difficulties. 
        It affects people of all ages and can vary in severity, from mild symptoms to life-threatening attacks.
        """)

        
        #caption="Representation of asthma and airways"
        
        st.header("Common Symptoms of Asthma")
        st.markdown("""
        - Shortness of breath  
        - Wheezing (a whistling sound when breathing)  
        - Chest tightness  
        - Coughing, especially at night or early morning  
        - Difficulty performing physical activities  
        """)


        st.header("Causes and Triggers")
        st.write("""
        Asthma is caused by a combination of genetic and environmental factors. Common triggers include:
        """)
        st.markdown("""
        - **Allergens:** Dust mites, pollen, mold, pet dander.  
        - **Irritants:** Smoke, pollution, strong odors, chemicals.  
        - **Weather Changes:** Cold air or sudden temperature changes.  
        - **Exercise-Induced Asthma:** Triggered by physical activity.  
        - **Infections:** Respiratory infections like colds or flu.  
        - **Stress or Anxiety:** Can exacerbate symptoms.  
        """)

        st.header("How is Asthma Managed?")
        st.write("""
        While asthma cannot be cured, it can be effectively managed through:
        """)
        st.markdown("""
        1. **Medications:**  
        - **Controller Medications:** Reduce inflammation and prevent symptoms (e.g., inhaled corticosteroids).  
        - **Rescue Inhalers:** Provide quick relief during asthma attacks (e.g., albuterol).  

        2. **Avoiding Triggers:** Identifying and minimizing exposure to asthma triggers.  

        3. **Monitoring Symptoms:** Keeping track of breathing patterns and using a peak flow meter.  

        4. **Action Plan:** Developing a personalized asthma action plan with a healthcare provider.  
        """)

        st.header("When to Seek Medical Attention")
        st.markdown("""
        You should seek immediate medical care if:
        - You experience severe difficulty breathing.  
        - Your rescue inhaler is not relieving symptoms.  
        - Your symptoms are worsening rapidly.  
        """)
        

    # Column 2: Causes, Management, and When to Seek Help
    with colA2:
        st.write("---")
        st.image("https://i.ibb.co/T2mtJzN/que-es-e-ingles.png",use_container_width=True)
        st.write("")
        st.write("---")
        st.write("")
        st.image("https://i.ibb.co/sjHWqhg/causas.png",use_container_width=True)
        st.write("")
        st.write("---")
        st.write("")
        st.image("https://i.ibb.co/mDFysk7/asthmasimpth.jpg",use_container_width=True)
        st.write("---")

    # Footer

    st.info("💡 Learn more about asthma by consulting your healthcare provider or visiting trusted medical websites like the WHO or CDC.")




    st.write("---")


    # Main title
    st.title("🤖 Introduction to Machine Learning")

     # Creating two columns
    colM1, colM2 = st.columns([2,1])

    with colM1:

        # Section: What is Machine Learning?
        st.header("What is Machine Learning?")
        st.write("""
        Machine Learning (ML) is a field of artificial intelligence that enables machines to learn from data 
        and improve their performance on specific tasks without being explicitly programmed. It uses algorithms 
        that identify patterns and relationships in the data, generating predictive models.
        """)

        # Illustrative image
        

        # Section: Types of Machine Learning
        st.header("Types of Machine Learning")
        st.markdown("""
        1. **Supervised Learning:**  
        Algorithms learn from labeled data.  
        *Example:* Classifying emails as "spam" or "not spam".

        2. **Unsupervised Learning:**  
        Finds patterns in unlabeled data.  
        *Example:* Customer segmentation in marketing.

        3. **Reinforcement Learning:**  
        Based on rewards and punishments to learn.  
        *Example:* Robots playing video games.
        """)

        # Section: Key Benefits of Machine Learning
        st.header("Key Benefits of Machine Learning")
        st.markdown("""
        1. **Process Automation:** Reduces human intervention in repetitive tasks.  
        2. **Improved Accuracy and Predictions:** Identifies complex patterns with high precision.  
        3. **Rapid Data Analysis:** Processes large amounts of data in real-time.  
        4. **Personalized Experiences:** Adapts services to each user's needs.  
        5. **Resource Optimization:** Reduces costs and improves operational efficiency.  
        6. **Fraud Detection:** Identifies anomalies in transactions and prevents fraud.  
        7. **Product Innovation:** Facilitates the development of new services like autonomous vehicles.  
        8. **Data-Driven Decision Making:** Helps make strategic decisions.  
        """)
    

    with colM2:

        st.image("https://i.ibb.co/8M1MR1Q/machinelearning.png",use_container_width=True)
        st.write("---")
        st.image("https://i.ibb.co/PcwYz9y/machiimg.jpg",use_container_width=True)
        st.write("---")
        st.image("https://i.ibb.co/xHy84JT/bussnies.jpg",use_container_width=True)


    # Footer
    st.info("💡 Want to learn more? Explore online courses on Machine Learning or research tools like TensorFlow and Scikit-Learn.")




if selected == "Diagnostic Measures":



    # Load the pre-trained model
    with open('models/3_Knn_asthma.pkl', 'rb') as file:
        model = pickle.load(file)

    st.title("Asthma Prediction Application")
    st.header("Predict Asthma Diagnosis Based on User Input")

    def get_user_input():
        Age = st.slider("Age (years)", 0, 80, 30)
        Gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
        BMI = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0)
        Smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
        PhysicalActivity = st.slider("Physical Activity (hours/week)", 0, 50, 5)
        DietQuality = st.slider("Diet Quality (0 = Poor, 10 = Excellent)", 0, 10, 5)
        SleepQuality = st.slider("Sleep Quality (0 = Poor, 10 = Excellent)", 0, 10, 7)
        PollutionExposure = st.slider("Pollution Exposure (0 = Low, 10 = High)", 0, 10, 3)
        PollenExposure = st.slider("Pollen Exposure (0 = Low, 10 = High)", 0, 10, 4)
        DustExposure = st.slider("Dust Exposure (0 = Low, 10 = High)", 0, 10, 3)
        PetAllergy = st.selectbox("Pet Allergy (0 = No, 1 = Yes)", [0, 1])
        FamilyHistoryAsthma = st.selectbox("Family History of Asthma (0 = No, 1 = Yes)", [0, 1])
        HistoryOfAllergies = st.selectbox("History of Allergies (0 = No, 1 = Yes)", [0, 1])
        Eczema = st.selectbox("Eczema (0 = No, 1 = Yes)", [0, 1])
        HayFever = st.selectbox("Hay Fever (0 = No, 1 = Yes)", [0, 1])
        GastroesophagealReflux = st.selectbox("Gastroesophageal Reflux (0 = No, 1 = Yes)", [0, 1])
        LungFunctionFEV1 = st.slider("Lung Function FEV1 (%)", 0.0, 100.0, 80.0)
        LungFunctionFVC = st.slider("Lung Function FVC (%)", 0.0, 100.0, 80.0)
        Wheezing = st.selectbox("Wheezing (0 = No, 1 = Yes)", [0, 1])
        ShortnessOfBreath = st.selectbox("Shortness of Breath (0 = No, 1 = Yes)", [0, 1])
        ChestTightness = st.selectbox("Chest Tightness (0 = No, 1 = Yes)", [0, 1])
        Coughing = st.selectbox("Coughing (0 = No, 1 = Yes)", [0, 1])
        NighttimeSymptoms = st.selectbox("Nighttime Symptoms (0 = No, 1 = Yes)", [0, 1])
        ExerciseInduced = st.selectbox("Exercise-Induced Symptoms (0 = No, 1 = Yes)", [0, 1])

        user_data = {
            'Age': Age, 'Gender': Gender, 'BMI': BMI, 'Smoking': Smoking,
            'PhysicalActivity': PhysicalActivity, 'DietQuality': DietQuality,
            'SleepQuality': SleepQuality, 'PollutionExposure': PollutionExposure,
            'PollenExposure': PollenExposure, 'DustExposure': DustExposure,
            'PetAllergy': PetAllergy, 'FamilyHistoryAsthma': FamilyHistoryAsthma,
            'HistoryOfAllergies': HistoryOfAllergies, 'Eczema': Eczema, 'HayFever': HayFever,
            'GastroesophagealReflux': GastroesophagealReflux, 'LungFunctionFEV1': LungFunctionFEV1,
            'LungFunctionFVC': LungFunctionFVC, 'Wheezing': Wheezing, 
            'ShortnessOfBreath': ShortnessOfBreath, 'ChestTightness': ChestTightness,
            'Coughing': Coughing, 'NighttimeSymptoms': NighttimeSymptoms,
            'ExerciseInduced': ExerciseInduced
        }

        return pd.DataFrame(user_data, index=[0])

    user_input = get_user_input()

    if st.button("Predict"):
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input)
        result = "Asthma Positive" if prediction[0] == 1 else "Asthma Negative"
        confidence = np.max(probability) * 100

        st.subheader("Prediction Result")
        st.success(f"Diagnosis: {result}")
        st.subheader("Prediction Confidence")
        st.info(f"{confidence:.2f}%")





if selected == "Evaluate Data":




    # Load the pre-trained model
    with open('models/3_Knn_asthma.pkl', 'rb') as file:
        model = pickle.load(file)

    st.title("Asthma Prediction Application")
    st.header("Evaluate Uploaded Data from Diagnostic Measures")

    # File uploader
    uploaded_file = st.file_uploader("Upload your data:", type=["csv"])

    if uploaded_file:
        st.subheader("Input Data")
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file, float_precision="round_trip")

        # Assuming the first columns of the dataset match the model's expected features
        feature_columns = [
            'Age', 'Gender', 'BMI', 'Smoking', 'PhysicalActivity', 
            'DietQuality', 'SleepQuality', 'PollutionExposure', 
            'PollenExposure', 'DustExposure', 'PetAllergy', 
            'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Eczema', 
            'HayFever', 'GastroesophagealReflux', 'LungFunctionFEV1', 
            'LungFunctionFVC', 'Wheezing', 'ShortnessOfBreath', 
            'ChestTightness', 'Coughing', 'NighttimeSymptoms', 
            'ExerciseInduced'
        ]

        # Select only the relevant features
        X = df[feature_columns].values

        # Make predictions and calculate probabilities
        prediction = model.predict(X)
        probability = model.predict_proba(X)

        # Prepare results
        pred = ["Asthma Negative" if i == 0 else "Asthma Positive" for i in prediction]
        negative_accuracy = [f"{(prob[0] * 100):.2f}%" for prob in probability]
        positive_accuracy = [f"{(prob[1] * 100):.2f}%" for prob in probability]

        # Add results to a DataFrame for display
        df_results = df.copy()
        df_results['Result'] = pred
        df_results['Asthma Negative Accuracy'] = negative_accuracy
        df_results['Asthma Positive Accuracy'] = positive_accuracy

        # Display the results
        st.write(df_results)

        # Provide download link for the results
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df_results)

        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='asthma_predictions.csv',
            mime='text/csv',
        )
