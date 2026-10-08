import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

# ==========================================
# CUSTOM TITLE
# ==========================================

st.title("💻 Laptop Price Prediction")
st.write("Enter the laptop specifications to predict the approximate price.")

st.divider()

# =========
# Image
# =========

st.image(
    "C:/Users/Admin/Downloads/Laptop Price Prediction(ML Project)/Laptop impage.png",
    use_container_width=True
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("C:/Users/Admin/Downloads/Laptop Price Prediction(ML Project)/Laptop_Price_Prediction_Model.pkt")
    return model

model = load_model()

# ==========================================
# CREATE COLUMNS
# ==========================================

col1, col2 = st.columns(2)

# ==========================================
# COLUMN 1
# ==========================================

with col1:

    product_name = st.selectbox(
        "Product Name",
        [
            "Dell Inspiron 15",
            "HP 15s",
            "Lenovo IdeaPad 3",
            "ASUS VivoBook 15",
            "Acer Aspire 5",
            "HP Pavilion 15",
            "Lenovo ThinkPad E14",
            "Dell Vostro 15"
        ]
    )

    brand = st.selectbox(
        "Brand",
        [
            "Dell",
            "HP",
            "Lenovo",
            "ASUS",
            "Acer",
            "MSI",
            "Apple",
            "Samsung",
            "Microsoft",
            "LG"
        ]
    )

    ram = st.selectbox(
        "RAM (GB)",
        [4, 8, 16, 32, 64]
    )

    storage = st.selectbox(
        "Storage (GB)",
        [256, 512, 1024, 2048]
    )

    processor = st.selectbox(
        "Processor",
        [
            "Intel Core i3",
            "Intel Core i5",
            "Intel Core i7",
            "Intel Core i9",
            "AMD Ryzen 3",
            "AMD Ryzen 5",
            "AMD Ryzen 7",
            "AMD Ryzen 9"
        ]
    )

# ==========================================
# COLUMN 2
# ==========================================

with col2:

    gpu = st.selectbox(
        "GPU",
        [
            "Intel UHD",
            "Intel Iris Xe",
            "AMD Radeon",
            "NVIDIA GTX 1650",
            "NVIDIA RTX 2050",
            "NVIDIA RTX 3050",
            "NVIDIA RTX 4060",
            "NVIDIA RTX 4070"
        ]
    )

    screen_size = st.selectbox(
        "Screen Size (Inches)",
        [13.3, 14.0, 15.6, 16.0, 17.3]
    )

    operating_system = st.selectbox(
        "Operating System",
        [
            "Windows 10",
            "Windows 11",
            "macOS",
            "Linux"
        ]
    )

    weight = st.number_input(
        "Weight (KG)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

if st.button(
    "🔮 Predict Laptop Price",
    use_container_width=True
):

    # ======================================
    # CREATE INPUT DATA
    # ======================================

    user_data = pd.DataFrame({
        "Product_Name": [product_name],
        "Brand": [brand],
        "RAM_GB": [ram],
        "Storage_GB": [storage],
        "Processor": [processor],
        "GPU": [gpu],
        "Screen_Size_Inches": [screen_size],
        "Operating_System": [operating_system],
        "Weight_KG": [weight]
    })

    # ======================================
    # PREDICTION
    # ======================================

    try:

        prediction = model.predict(user_data)

        predicted_price = prediction[0]

        st.success("✅ Prediction Successful!")

        st.metric(
            label="💰 Predicted Laptop Price",
            value=f"₹{predicted_price:,.2f}"
        )

        # ==================================
        # SHOW USER INPUT
        # ==================================

        st.subheader("📋 Laptop Details")

        display_data = pd.DataFrame({
            "Specification": [
                "Product Name",
                "Brand",
                "RAM",
                "Storage",
                "Processor",
                "GPU",
                "Screen Size",
                "Operating System",
                "Weight"
            ],

            "Value": [
                product_name,
                brand,
                f"{ram} GB",
                f"{storage} GB",
                processor,
                gpu,
                f"{screen_size} Inches",
                operating_system,
                f"{weight} KG"
            ]
        })

        st.table(display_data)

    except Exception as e:

        st.error("❌ Prediction Error")

        st.write("Error Details:")
        st.code(str(e))

        st.info(
            "The input column names or values may not match "
            "the columns used while training the model."
        )