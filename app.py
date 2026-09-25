import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
from groq import Groq

st.set_page_config(page_title="BaytIQ", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("BaytIQ: Inclusive AI Real Estate Matcher")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

LOCATION_COORDS = {
    'Zamalek': [30.0626, 31.2223], 'Maadi': [29.9592, 31.2590], 'New Cairo': [30.0300, 31.4700],
    'Sheikh Zayed': [30.0400, 30.9800], 'Nasr City': [30.0600, 31.3300], 'Heliopolis': [30.1000, 31.3300],
    'Dokki': [30.0380, 31.2110], 'Mohandeseen': [30.0500, 31.2000], '6th of October': [29.9333, 30.9167],
    'Madinaty': [30.0900, 31.6200], 'Al Rehab': [30.0600, 31.4900], 'El Shorouk': [30.1400, 31.6200],
    'El Obour': [30.2200, 31.4700], 'Mokattam': [30.0100, 31.3000], 'Downtown Cairo': [30.0444, 31.2357],
    'Shoubra': [30.0700, 31.2400], 'Helwan': [29.8400, 31.3000], 'Haram': [29.9800, 31.1300],
    'Faisal': [30.0000, 31.1500], 'New Capital': [29.9800, 31.7200]
}

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model('inclusive_model.cbm')
    return model

@st.cache_data
def load_data():
    return pd.read_csv('inclusive_real_estate_data.csv')

model = load_model()
df_houses = load_data()

def find_best_match(user_profile, df_houses):
    max_budget = user_profile['User_Budget'] + 1000
    affordable = df_houses[df_houses['Monthly_Rent_EGP'] <= max_budget].copy()
    
    if affordable.empty:
        return None
    
    eval_df = pd.DataFrame({
        'User_Budget': user_profile['User_Budget'],
        'User_Wheelchair': user_profile['User_Wheelchair'],
        'User_Elderly': user_profile['User_Elderly'],
        'User_Respiratory': user_profile['User_Respiratory'],
        'User_Neurodivergent': user_profile['User_Neurodivergent'],
        'House_Rent': affordable['Monthly_Rent_EGP'],
        'House_Neighborhood': affordable['Neighborhood'],
        'House_Wheelchair': affordable['Wheelchair_Accessible'],
        'House_Elevator': affordable['Elevator_Access'],
        'House_Air_Quality': affordable['Air_Quality_Index'],
        'House_Medical': affordable['Medical_Proximity_Score'],
        'House_Quietness': affordable['Quietness_Score'],
        'House_Soundproof': affordable['Soundproofing_Score'],
        'House_Light': affordable['Natural_Light_Index']
    })
    
    affordable['Match_Score'] = model.predict(eval_df)
    return affordable.sort_values(by='Match_Score', ascending=False).iloc[0]

def generate_ai_blueprint(user_profile, house):
    profiles = []
    if user_profile['User_Wheelchair']: profiles.append("Wheelchair User")
    if user_profile['User_Elderly']: profiles.append("Elderly")
    if user_profile['User_Respiratory']: profiles.append("Respiratory Issues")
    if user_profile['User_Neurodivergent']: profiles.append("Neurodivergent")
    
    profile_str = ", ".join(profiles) if profiles else "Standard Lifestyle"
    
    prompt = f"You are an AI Interior Design Expert. The user profile is: {profile_str}. The matched house is in {house['Neighborhood']} with Air Quality: {house['Air_Quality_Index']}/10, Soundproofing: {house['Soundproofing_Score']}/10, Light: {house['Natural_Light_Index']}/10. Write a strict 3-sentence actionable interior design and spatial arrangement blueprint tailored to their specific medical or psychological needs."
    
    try:
        response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except:
        return "Ensure optimal furniture placement for accessibility and use ambient lighting to enhance comfort."

def chat_with_agent(question, house):
    prompt = f"""You are AqarBot, an AI Real Estate Assistant. 
    Answer in English based on these details:
    Location: {house['Neighborhood']}, Rent: {house['Monthly_Rent_EGP']} EGP, Wheelchair Accessible: {bool(house['Wheelchair_Accessible'])}, Elevator: {bool(house['Elevator_Access'])}, Air Quality: {house['Air_Quality_Index']}/10, Medical Proximity: {house['Medical_Proximity_Score']}/10, Soundproof: {house['Soundproofing_Score']}/10, Light: {house['Natural_Light_Index']}/10.
    Question: {question}"""
    try:
        response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}], temperature=0.7)
        return response.choices[0].message.content
    except:
        return "Network issue. Please try again."

st.sidebar.header("User Profile")
budget = st.sidebar.number_input("Monthly Budget (EGP)", 4000, 50000, 15000, step=1000)

st.sidebar.subheader("Health & Accessibility Needs")
wheelchair = st.sidebar.checkbox("Wheelchair Access Required")
elderly = st.sidebar.checkbox("Elderly / Limited Mobility")
respiratory = st.sidebar.checkbox("Respiratory Issues (Asthma/Allergies)")
neurodivergent = st.sidebar.checkbox("Neurodivergent (Autism/ADHD)")

user_profile = {
    'User_Budget': budget,
    'User_Wheelchair': int(wheelchair),
    'User_Elderly': int(elderly),
    'User_Respiratory': int(respiratory),
    'User_Neurodivergent': int(neurodivergent)
}

if st.sidebar.button("Find Inclusive Match", use_container_width=True):
    with st.spinner("Analyzing data and generating blueprint..."):
        best_match = find_best_match(user_profile, df_houses)
        
        if best_match is not None:
            st.session_state['best_match'] = best_match
            score = min(100.0, max(0.0, best_match['Match_Score']))
            
            st.success(f"Optimal Match Found in: **{best_match['Neighborhood']}**")
            st.progress(int(score) / 100.0, text=f"Match Score: {score:.1f}%")
            
            lat_lon = LOCATION_COORDS.get(best_match['Neighborhood'], [30.0444, 31.2357])
            df_map = pd.DataFrame({'lat': [lat_lon[0]], 'lon': [lat_lon[1]]})
            st.map(df_map, zoom=11)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rent (EGP)", f"{best_match['Monthly_Rent_EGP']}")
            col2.metric("Bedrooms", f"{best_match['Bedrooms']}")
            col3.metric("Air Quality", f"{best_match['Air_Quality_Index']}/10")
            col4.metric("Medical Proximity", f"{best_match['Medical_Proximity_Score']}/10")
            
            st.divider()
            st.subheader("AI Medical & Interior Blueprint")
            blueprint = generate_ai_blueprint(user_profile, best_match)
            st.info(blueprint)
            
            st.download_button(
                label="Download Blueprint as TXT",
                data=blueprint,
                file_name=f"BaytIQ_{best_match['Neighborhood']}_Blueprint.txt",
                mime="text/plain",
                use_container_width=True
            )
                
        else:
            st.error("No properties found within this budget.")

if 'best_match' in st.session_state:
    st.divider()
    st.subheader("AqarBot Assistant")
    user_q = st.text_input("Ask AqarBot about this property:")
    if st.button("Ask") and user_q:
        with st.spinner("Processing..."):
            answer = chat_with_agent(user_q, st.session_state['best_match'])
            st.success(answer)
