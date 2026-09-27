import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
from groq import Groq

st.set_page_config(page_title="BaytIQ", layout="wide", page_icon="🏠")

st.title(" BaytIQ: Inclusive AI Real Estate Matcher")
st.markdown("Discover properties perfectly matched to your **medical, accessibility, and lifestyle needs** using ML & Generative AI.")

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

def find_top_matches(user_profile, df_houses, top_n=3):
    max_budget = user_profile['User_Budget'] + 1000
    valid_houses = df_houses[df_houses['Monthly_Rent_EGP'] <= max_budget].copy()
    
    if user_profile['User_Wheelchair']:
        valid_houses = valid_houses[valid_houses['Wheelchair_Accessible'] == 1]
    if user_profile['User_Elderly']:
        valid_houses = valid_houses[valid_houses['Elevator_Access'] == 1]
        
    if valid_houses.empty:
        return None
    
    eval_df = pd.DataFrame({
        'User_Budget': user_profile['User_Budget'],
        'User_Wheelchair': user_profile['User_Wheelchair'],
        'User_Elderly': user_profile['User_Elderly'],
        'User_Respiratory': user_profile['User_Respiratory'],
        'User_Neurodivergent': user_profile['User_Neurodivergent'],
        'House_Rent': valid_houses['Monthly_Rent_EGP'],
        'House_Neighborhood': valid_houses['Neighborhood'],
        'House_Wheelchair': valid_houses['Wheelchair_Accessible'],
        'House_Elevator': valid_houses['Elevator_Access'],
        'House_Air_Quality': valid_houses['Air_Quality_Index'],
        'House_Medical': valid_houses['Medical_Proximity_Score'],
        'House_Quietness': valid_houses['Quietness_Score'],
        'House_Soundproof': valid_houses['Soundproofing_Score'],
        'House_Light': valid_houses['Natural_Light_Index']
    })
    
    valid_houses['Match_Score'] = model.predict(eval_df).clip(0, 100)
    return valid_houses.sort_values(by='Match_Score', ascending=False).head(top_n)

def explain_match(house, user_profile):
    positives, negatives = [], []
    
    if user_profile['User_Wheelchair'] and house['Wheelchair_Accessible']:
        positives.append("Fully wheelchair accessible (Meets your strict requirement).")
    if user_profile['User_Elderly'] and house['Elevator_Access']:
        positives.append("Elevator access available (Meets your strict requirement).")
        
    if house['Air_Quality_Index'] >= 8:
        positives.append(f"Excellent air quality ({house['Air_Quality_Index']}/10).")
    elif house['Air_Quality_Index'] <= 5 and user_profile['User_Respiratory']:
        negatives.append(f"Moderate air quality ({house['Air_Quality_Index']}/10) - Air purifiers recommended.")
        
    if house['Medical_Proximity_Score'] >= 7:
        positives.append(f"Close to medical facilities ({house['Medical_Proximity_Score']}/10).")
        
    if house['Monthly_Rent_EGP'] > user_profile['User_Budget']:
        negatives.append(f"Rent exceeds your base budget by {house['Monthly_Rent_EGP'] - user_profile['User_Budget']} EGP.")
    else:
        positives.append("Completely within your budget.")
        
    return positives, negatives

def generate_ai_blueprint(user_profile, house):
    profiles = []
    if user_profile['User_Wheelchair']: profiles.append("Wheelchair User")
    if user_profile['User_Elderly']: profiles.append("Elderly")
    if user_profile['User_Respiratory']: profiles.append("Respiratory Issues")
    if user_profile['User_Neurodivergent']: profiles.append("Neurodivergent")
    
    profile_str = ", ".join(profiles) if profiles else "Standard Lifestyle"
    
    prompt = f"You are an AI Accessibility & Interior Design Expert. User profile: {profile_str}. House in {house['Neighborhood']}. Write a very concise, 3-bullet-point interior design blueprint tailored to their accessibility needs. Keep each bullet short, punchy, and highly readable. Do not give medical advice."
    
    try:
        response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except:
        return "• Ensure optimal furniture placement for accessibility.\n• Use ambient lighting to enhance comfort.\n• Keep pathways clear."

def chat_with_agent(question, house):
    prompt = f"You are AqarBot. Answer using ONLY these details: Location: {house['Neighborhood']}, Rent: {house['Monthly_Rent_EGP']} EGP, Wheelchair: {bool(house['Wheelchair_Accessible'])}, Elevator: {bool(house['Elevator_Access'])}, Air Quality: {house['Air_Quality_Index']}/10. Question: {question}"
    try:
        response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}], temperature=0.5)
        return response.choices[0].message.content
    except:
        return "Network issue. Please try again."

st.sidebar.header(" User Profile")
budget = st.sidebar.number_input("Monthly Budget (EGP)", 4000, 50000, 15000, step=1000)

st.sidebar.subheader(" Health & Accessibility")
wheelchair = st.sidebar.checkbox("Wheelchair Access (Hard Constraint)")
elderly = st.sidebar.checkbox("Elderly / Elevator (Hard Constraint)")
respiratory = st.sidebar.checkbox("Respiratory Issues (Asthma/Allergies)")
neurodivergent = st.sidebar.checkbox("Neurodivergent (Autism/ADHD)")

st.sidebar.divider()
st.sidebar.subheader(" How BaytIQ Works")
st.sidebar.info("1. **Hard Constraints:** Filters properties lacking required physical accessibility.\n2. **CatBoost Engine:** Ranks properties on 14 environmental features.\n3. **Groq LLM:** Generates personalized accessibility blueprints.")

user_profile = {
    'User_Budget': budget, 'User_Wheelchair': int(wheelchair),
    'User_Elderly': int(elderly), 'User_Respiratory': int(respiratory),
    'User_Neurodivergent': int(neurodivergent)
}

if st.sidebar.button(" Find Top Matches", use_container_width=True, type="primary"):
    with st.spinner("Analyzing data & generating AI blueprints..."):
        top_matches = find_top_matches(user_profile, df_houses)
        
        if top_matches is not None and not top_matches.empty:
            st.success(f" Found {len(top_matches)} highly compatible properties!")
            
            tab_titles = [f"#{i+1}: {row['Neighborhood']} ({row['Match_Score']:.1f}%)" for i, row in top_matches.reset_index().iterrows()]
            tabs = st.tabs(tab_titles)
            
            for i, tab in enumerate(tabs):
                house = top_matches.iloc[i]
                score = house['Match_Score']
                
                with tab:
                    st.progress(int(score) / 100.0, text=f"AI Match Score: {score:.1f}%")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Rent (EGP)", f"{house['Monthly_Rent_EGP']}")
                    col2.metric("Bedrooms", f"{house['Bedrooms']}")
                    col3.metric("Air Quality", f"{house['Air_Quality_Index']}/10")
                    col4.metric("Medical", f"{house['Medical_Proximity_Score']}/10")
                    
                    col_map, col_explain = st.columns([1, 1])
                    with col_map:
                        lat_lon = LOCATION_COORDS.get(house['Neighborhood'], [30.0444, 31.2357])
                        st.map(pd.DataFrame({'lat': [lat_lon[0]], 'lon': [lat_lon[1]]}), zoom=11)
                    
                    with col_explain:
                        st.markdown("####  Why this property?")
                        pos, neg = explain_match(house, user_profile)
                        for p in pos:
                            st.success(p)
                        if neg:
                            for n in neg:
                                st.warning(n)
                    
                    st.divider()
                    st.subheader(" AI Accessibility & Interior Blueprint")
                    blueprint = generate_ai_blueprint(user_profile, house)
                    st.info(blueprint)
                    
                    st.download_button(label=f" Download Blueprint", data=blueprint, file_name=f"BaytIQ_Blueprint_{i+1}.txt", mime="text/plain", key=f"dl_{i}")
                    
                    with st.expander(" Ask AqarBot about this property"):
                        user_q = st.text_input("What would you like to know?", key=f"q_{i}")
                        if st.button("Ask", key=f"btn_{i}") and user_q:
                            st.success(chat_with_agent(user_q, house))
        else:
            st.error("No properties found matching your strict hard constraints and budget.")
