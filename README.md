# AgroScan AI - Andean Crop Phytosanitary Engine 🌾🔬

**Production-grade FastAPI backend for Andean crop disease diagnosis, real-time Supabase PostgreSQL persistence, and interactive agronomic AI follow-up consultation.**

Developed specifically for smallholder farming communities in **Nariño, Colombia** (elevations 1,500m – 3,200m a.s.l.), focusing on **Potato** (*Solanum tuberosum*), **Coffee** (*Coffea arabica*), **Corn** (*Zea mays*), and **Tomato** (*Solanum lycopersicum*).

---

## 1. Architectural Overview

```
┌────────────────────────────────────────────────────────┐
│                   Client Applications                  │
│       (Web Frontend / Mobile / Agricultural Workers)   │
└───────────────────────────┬────────────────────────────┘
                            │ Multipart/Form-Data & JSON
                            ▼
┌────────────────────────────────────────────────────────┐
│                 FastAPI REST Backend                   │
│                    (Python 3.11+)                      │
│   • Multimodal image preprocessing                     │
│   • CORS middleware for web clients                    │
│   • Asynchronous lifespan schema initialization       │
└─────────────┬────────────────────────────┬─────────────┘
              │ Multimodal Image Parts     │ SQLAlchemy +
              │ & Structured Output Schema │ psycopg2-binary
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│     Google Gemini 2.5     │  │   Supabase PostgreSQL   │
│        Flash Vision       │  │    (PgBouncer Pooler)   │
│ • Disease identification  │  │ • diagnostics table     │
│ • Severity classification │  │ • chat_thread_messages  │
│ • Organic & chemical plans│  │ • Connection pooling    │
└───────────────────────────┘  └─────────────────────────┘
```

---

## 2. Key Features

- **JWT Authentication & Farmer Accounts**: Secure registration and login using bcrypt password hashing, issuing signed HS256 JWT tokens to associate diagnostic records and history with specific farmers.
- **Multimodal Visual Diagnosis**: Direct plant symptom analysis with Google Gemini 2.5 Flash, returning structured phytopathological reports.
- **Relational Persistence**: Cloud-hosted PostgreSQL managed by **Supabase** via PgBouncer transaction pooling (`port 6543`, `pool_pre_ping=True`, `pool_recycle=300`).
- **Interactive Follow-Up Consultation**: Grounded agronomic advisory engine calculating calibrated dosages for standard **20-Liter agricultural backpack sprayers** (*bombas de espalda de 20L*), rainfastness thresholds, and required personal protective equipment (PPE).
- **Graceful Offline Fallback**: Resilient fallback engine capable of serving Andean agronomic guidelines even during network or API disruptions.

---

## 3. Database Schema (Supabase PostgreSQL)

### `users` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `VARCHAR(64)` | Primary Key (UUID format `usr-xxx`) |
| `full_name` | `VARCHAR(255)` | Farmer or agronomist full name |
| `email` | `VARCHAR(255)` | Unique, indexed email address |
| `hashed_password` | `VARCHAR(255)` | One-way salted bcrypt password hash |
| `farm_name` | `VARCHAR(255)` | Farm parcel or finca designation |
| `created_at` | `TIMESTAMPTZ` | Registration timestamp (UTC) |

### `diagnostics` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `VARCHAR(64)` | Primary Key (UUID format `diag-xxx`) |
| `user_id` | `VARCHAR(64)` | Foreign Key -> `users(id)` (ON DELETE SET NULL, Indexed) |
| `crop_type` | `VARCHAR(64)` | Crop variety (`Potato`, `Coffee`, `Corn`, `Tomato`) |
| `plot_name` | `VARCHAR(128)` | Farm parcel identifier (Indexed) |
| `disease_name` | `VARCHAR(255)` | Identified condition or pest |
| `scientific_name` | `VARCHAR(255)` | Binomial taxonomic name |
| `pathogen_type` | `VARCHAR(64)` | `FUNGUS`, `BACTERIA`, `VIRUS`, `PEST`, or `HEALTHY` |
| `confidence_score` | `FLOAT` | Diagnostic confidence score (0.0 to 1.0) |
| `severity_level` | `VARCHAR(32)` | `LOW`, `MODERATE`, or `CRITICAL` |
| `symptoms_analysis`| `TEXT` | Detailed agronomic visual analysis |
| `organic_treatment`| `JSON` | List of organic inputs and biological controls |
| `chemical_treatment`| `JSON` | List of conventional chemical active ingredients |
| `preventive_actions`| `JSON` | Cultural and sanitation preventive practices |
| `disclaimer` | `VARCHAR(500)` | Agronomic advisory disclaimer |
| `timestamp` | `TIMESTAMPTZ` | Record creation timestamp (UTC) |

### `chat_thread_messages` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `SERIAL` | Primary Key (Autoincrementing integer) |
| `diagnostic_id` | `VARCHAR(64)` | Foreign Key -> `diagnostics(id)` (ON DELETE CASCADE) |
| `role` | `VARCHAR(32)` | `"user"` or `"model"` |
| `content` | `TEXT` | Message text content |
| `timestamp` | `TIMESTAMPTZ` | Timestamp of message turn (UTC) |

### `dosage_assessments` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `VARCHAR(64)` | Primary Key (UUID format `dose-xxx`) |
| `diagnostic_id` | `VARCHAR(64)` | Foreign Key -> `diagnostics(id)` (ON DELETE CASCADE, Indexed) |
| `plot_area_sqm` | `FLOAT` | Evaluated parcel area in square meters |
| `total_tanks_needed` | `FLOAT` | Number of 20-Liter backpack spray loads required |
| `product_recommended` | `VARCHAR(255)` | Recommended active ingredient or biological bio-input |
| `concentration_per_tank`| `VARCHAR(255)` | Grams or milliliters per 20L tank |
| `rain_safety_interval_hours`| `INTEGER` | Minimum rainfast drying interval (hours) |
| `economic_loss_risk` | `FLOAT` | Estimated yield loss percentage if left untreated within 5 days |
| `full_prescription_payload` | `JSON` | Full prescription JSON including PPE and application tips |
| `created_at` | `TIMESTAMPTZ` | Assessment creation timestamp (UTC) |

---

## 4. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root service metadata and API capabilities |
| `GET` | `/health` | Live health probe testing Supabase connection & Gemini setup |
| `POST` | `/api/v1/auth/register` | Register new farmer account and issue JWT access token |
| `POST` | `/api/v1/auth/login` | Authenticate farmer with bcrypt hash and return JWT |
| `GET` | `/api/v1/auth/me` | Retrieve authenticated farmer profile (`Bearer <token>`) |
| `POST` | `/api/v1/diagnose` | Upload crop leaf image (`multipart/form-data`) for AI diagnosis |
| `GET` | `/api/v1/history` | Retrieve historical diagnosis records (filtered by user if authenticated) |
| `POST` | `/api/v1/diagnose/{id}/chat` | Interactive agronomic follow-up consultation grounded on diagnosis |
| `POST` | `/api/v1/diagnose/{id}/dosage` | Calculate calibrated 20L backpack tank requirements & weather risk |
| `GET` | `/api/v1/diagnose/{id}/dosage` | Retrieve previously calculated dosage assessments for a diagnosis |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `GET` | `/redoc` | OpenAPI Redoc documentation |

---

## 5. AI Backpack Sprayer Calibration & Andean Weather Risk Engine

Smallholder Andean farmers in Nariño cultivate steep hillside terraces where precision spraying and weather timing are critical to avoid chemical wash-off and yield loss.

### Agronomic Calibration Heuristics:
- **Application Volume per Hectare**:
  * **Potato (*Solanum tuberosum*) & Tomato (*Solanum lycopersicum*)**: $250 - 400\text{ L/ha}$ (dense canopy coverage, equal to 12.5 to 20 tanks of 20L per hectare).
  * **Coffee (*Coffea arabica*) & Corn (*Zea mays*)**: $150 - 300\text{ L/ha}$ (7.5 to 15 tanks of 20L per hectare).
- **Backpack Sprayer Sizing Formulation**:
  $$\text{Tanks Needed} = \max\left(1.0,\, \frac{\text{Plot Area (m}^2\text{)}}{10,000} \times \frac{\text{Target Volume (L/ha)}}{20\text{ L/tank}}\right)$$
- **Rainfastness Safety Interval**:
  Andean afternoon convective rain cycles require a minimum drying window of 3 to 4 hours post-application for systemic fungicides (or 2 hours with organosilicone surfactants).
- **Economic Loss Risk**:
  Quantifies potential 5-day crop loss based on pathogen virulence (e.g., *Phytophthora infestans* Late Blight can destroy potato canopies in 7-10 days under humid conditions).

---

## 6. Environment Variables Setup

Create a `.env` file in the project root (see `.env.example`):

```env
# Google Gemini API Key (Obtain from Google AI Studio: https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here

# Supabase PostgreSQL Pooler URI (PgBouncer)
DATABASE_URL=postgresql://postgres.dthmcpvbbhqvbywpjpre:3GVS?rqGD%zBGy@@aws-0-us-east-2.pooler.supabase.com:6543/postgres?pgbouncer=true

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Allowed CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:5173,http://localhost:5180
```

---

## 7. Installation & Execution

### 1. Create and Activate Virtual Environment
```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
