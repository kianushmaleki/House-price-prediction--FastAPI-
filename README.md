# 🏠 House Price Prediction API Service

This project bridges the gap between Data Science and Software Engineering by deploying a trained house price prediction model as a production-ready **RESTful API**. Built with **FastAPI**, this service provides high-performance inference, automated validation, and comprehensive system monitoring.



---

## 🏗️ System Architecture
The application follows a modular "Separation of Concerns" design to ensure the code is maintainable and testable.

* **`schemas.py` (The Contract)**: Utilizes **Pydantic** for rigorous data validation. It ensures that only valid, cleaned data reaches the model and that API responses remain consistent.
* **`api.py` (The Engine)**: Contains the core business logic. This layer handles file I/O for `model.pkl`, converts JSON inputs into **NumPy** arrays, and executes the prediction logic.
* **`main.py` (The Gateway)**: Acts as the entry point. It manages the **Uvicorn** server lifecycle, configures routes, and ensures the model is loaded into memory only once during the `startup` event for maximum efficiency.

---

## 🛠️ Key Features
* **⚡ Real-Time Inference**: A `POST /predict` endpoint that accepts 13 house features and returns a numerical price estimate instantly.
* **📖 Self-Documenting**: Full integration with **Swagger UI** (accessible via `/docs`) and **ReDoc**, allowing for interactive testing without external tools like Postman.
* **🩺 Health Monitoring**: A dedicated `/health` endpoint to verify service uptime and confirm that the machine learning model is correctly loaded.
* **🔍 Model Transparency**: The `/model/info` endpoint exposes critical metadata, including model version, training date, and RMSE performance metrics.

---

## 🚀 Getting Started

### 📋 Prerequisites
* Python 3.8+
* FastAPI & Uvicorn
* Scikit-Learn & NumPy
* Pydantic

### ⚙️ Installation & Execution
1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/house-price-api.git](https://github.com/your-username/house-price-api.git)
    cd house-price-api
    ```

2.  **Install Dependencies**:
    ```bash
    pip install fastapi uvicorn pandas scikit-learn pydantic
    ```

3.  **Launch the Server**:
    ```bash
    uvicorn main:app --reload
    ```

4.  **Explore the API**:
    * **Root API**: `http://127.0.0.1:8000`
    * **Interactive Documentation**: `http://127.0.0.1:8000/docs`

---

## 📊 API Documentation

| Endpoint | Method | Input | Output | Description |
| :--- | :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | `HealthCheckResponse` | Verifies service and model status. |
| `/model/info` | `GET` | None | `ModelInfoResponse` | Returns model version and performance metadata. |
| `/predict` | `POST` | `HouseFeatures` | `PredictionResponse` | Returns the predicted market value for a property. |

---

## ✅ Deployment Checklist
- [ ] Model and Metadata files are present in the root directory.
- [ ] `startup_event` successfully loads the model into global state.
- [ ] Input validation correctly rejects malformed JSON.
- [ ] All Pydantic models include field descriptions and examples.
- [ ] Error handling (404/500) is implemented for model loading failures.
