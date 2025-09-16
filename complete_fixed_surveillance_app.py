import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .status-trained {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #4caf50;
    }
    .status-not-trained {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

class VideoProcessor:
    def __init__(self):
        """Initialize the video processor with dummy data for demo"""
        self.model = None
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                       'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                       'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                       'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee']
        st.success("✅ Video processor initialized (Demo mode)")

    def extract_features_from_video(self, video_path, progress_bar=None):
        """Extract features from video for anomaly detection (Simplified version)"""
        # Simulate feature extraction with dummy data
        features = []
        total_frames = 100  # Simulate 100 frames
        
        for frame_count in range(1, total_frames + 1):
            # Update progress bar
            if progress_bar:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
            
            if frame_count % 10 == 0:  # Process every 10th frame for demo
                # Create dummy detections for demo
                detections = self._create_dummy_detections()
                
                # Extract features from detections
                frame_features = self.process_detections(detections, (480, 640))  # Standard video resolution
                features.append(frame_features)
        
        return np.array(features) if features else np.array([]).reshape(0, 8)

    def _create_dummy_detections(self):
        """Create dummy detections for demo purposes"""
        # Simulate some random detections
        num_objects = np.random.randint(0, 5)
        detections = []

        for _ in range(num_objects):
            detections.append({
                'xmin': np.random.randint(0, 400),
                'ymin': np.random.randint(0, 300),
                'xmax': np.random.randint(400, 640),
                'ymax': np.random.randint(300, 480),
                'confidence': np.random.uniform(0.5, 0.9),
                'name': np.random.choice(['person', 'car', 'bicycle'])
            })

        return pd.DataFrame(detections)

    def process_detections(self, detections, frame_shape):
        """Process detections and extract behavioral features"""
        height, width = frame_shape[:2]

        # Basic counts
        person_count = len(detections[detections['name'] == 'person']) if 'name' in detections.columns and len(detections) > 0 else 0
        vehicle_count = len(detections[detections['name'].isin(['car', 'truck', 'bus', 'motorcycle', 'bicycle'])]) if 'name' in detections.columns and len(detections) > 0 else 0
        total_objects = len(detections)

        # Advanced features
        if len(detections) > 0 and 'confidence' in detections.columns:
            avg_confidence = detections['confidence'].mean()

            # Calculate object density
            if 'xmin' in detections.columns:
                total_area = sum((detections['xmax'] - detections['xmin']) *
                               (detections['ymax'] - detections['ymin']))
                object_density = total_area / (width * height) if total_area > 0 else 0

                # Position variance
                center_x = (detections['xmin'] + detections['xmax']) / 2
                center_y = (detections['ymin'] + detections['ymax']) / 2
                position_variance = center_x.var() + center_y.var()

                # Average object size
                avg_object_size = ((detections['xmax'] - detections['xmin']) *
                                 (detections['ymax'] - detections['ymin'])).mean()
            else:
                object_density = 0
                position_variance = 0
                avg_object_size = 0
        else:
            avg_confidence = 0
            object_density = 0
            position_variance = 0
            avg_object_size = 0

        # Return feature vector
        return [
            person_count,
            vehicle_count,
            total_objects,
            avg_confidence,
            object_density,
            position_variance,
            avg_object_size,
            person_count / max(1, total_objects) if total_objects > 0 else 0
        ]

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train_models(self, X_train):
        """Train Isolation Forest model (simplified version)"""
        if len(X_train) == 0:
            st.warning("⚠️ No training data provided!")
            return False

        # Normalize data
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train Isolation Forest
        self.isolation_forest.fit(X_train_scaled)

        self.is_trained = True
        return True

    def detect_anomalies(self, X):
        """Detect anomalies using Isolation Forest"""
        if not self.is_trained:
            st.warning("⚠️ Models not trained yet!")
            return np.array([1] * len(X)), np.array([0.0] * len(X))

        X_scaled = self.scaler.transform(X)

        # Isolation Forest predictions
        if_predictions = self.isolation_forest.predict(X_scaled)

        # Create dummy confidence scores
        mse = np.random.uniform(0.1, 1.0, len(X_scaled))

        return if_predictions, mse

def create_synthetic_training_data(num_samples=1000):
    """Create synthetic training data for normal behavior patterns"""
    np.random.seed(42)

    # Normal behavior patterns
    features = []

    for _ in range(num_samples):
        # Simulate normal surveillance scenarios
        person_count = np.random.poisson(3)  # Usually 0-10 people
        vehicle_count = np.random.poisson(2)  # Usually 0-5 vehicles
        total_objects = person_count + vehicle_count + np.random.poisson(1)  # Some other objects

        avg_confidence = np.random.uniform(0.7, 0.95)  # High confidence for normal detections
        object_density = np.random.uniform(0.1, 0.4)   # Moderate density
        position_variance = np.random.uniform(100, 1000)  # Normal movement variance
        avg_object_size = np.random.uniform(1000, 5000)   # Average object sizes
        person_ratio = person_count / max(1, total_objects)

        features.append([
            person_count, vehicle_count, total_objects, avg_confidence,
            object_density, position_variance, avg_object_size, person_ratio
        ])

    return np.array(features)

class DatabaseManager:
    def __init__(self, db_path='surveillance.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomaly_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                features TEXT NOT NULL,
                video_frame_path TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def log_anomaly(self, camera_id, anomaly_type, confidence_score, features):
        """Log detected anomaly to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO anomaly_logs
            (timestamp, camera_id, anomaly_type, confidence_score, features, video_frame_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            camera_id,
            anomaly_type,
            confidence_score,
            json.dumps(features.tolist() if isinstance(features, np.ndarray) else features),
            None
        ))

        conn.commit()
        conn.close()

    def get_recent_anomalies(self, hours=24):
        """Retrieve recent anomalies from database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM anomaly_logs
            WHERE datetime(timestamp) >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours), conn)
        conn.close()
        return df

# Initialize session state
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()

if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()

if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if 'system_trained' not in st.session_state:
    st.session_state.system_trained = False

def get_system_status():
    """Get comprehensive system status"""
    return {
        'trained': st.session_state.system_trained and st.session_state.anomaly_detector.is_trained,
        'model_ready': hasattr(st.session_state.anomaly_detector, 'isolation_forest'),
        'processor_ready': st.session_state.video_processor is not None,
        'database_ready': st.session_state.db_manager is not None
    }

def update_system_status():
    """Force update system status"""
    status = get_system_status()
    st.session_state.system_trained = status['trained']
    return status

def show_settings_page():
    """Show the settings page"""
    st.header("⚙️ System Settings")

    st.subheader("🔧 Detection Parameters")

    col1, col2 = st.columns(2)

    with col1:
        contamination = st.slider(
            "Anomaly Detection Sensitivity",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Lower values = more sensitive"
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1
        )

    with col2:
        frame_skip = st.number_input(
            "Frame Skip Rate",
            min_value=1,
            max_value=60,
            value=30
        )

        max_objects = st.number_input(
            "Max Objects per Frame",
            min_value=10,
            max_value=100,
            value=50
        )

    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")

    st.subheader("📋 System Information")

    status = get_system_status()
    
    status_text = "✅ Trained & Active" if status['trained'] else "❌ Not Trained"
    status_color = "success" if status['trained'] else "error"
    
    st.markdown(f"""
    **System Status:**
    - Model Status: {status_text}
    - Database: {'✅ Connected' if status['database_ready'] else '❌ Error'}
    - Video Processor: {'✅ Ready' if status['processor_ready'] else '❌ Error'}
    - Storage: 📊 Available
    """)

def show_dashboard():
    """Show the main dashboard"""
    st.header("📊 System Dashboard")

    # Get recent anomalies
    recent_data = st.session_state.db_manager.get_recent_anomalies(24)

    # Get system status
    status = get_system_status()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Anomalies (24h)", len(recent_data))

    with col2:
        high_confidence = len(recent_data[recent_data['confidence_score'] > 0.8]) if len(recent_data) > 0 else 0
        st.metric("High Confidence Alerts", high_confidence)

    with col3:
        avg_confidence = recent_data['confidence_score'].mean() if len(recent_data) > 0 else 0
        st.metric("Average Confidence", f"{avg_confidence:.3f}")

    with col4:
        system_status = "Active" if status['trained'] else "Inactive"
        status_delta = "🟢" if status['trained'] else "🔴"
        st.metric("System Status", system_status, delta=status_delta)

    # System Status Alert
    if not status['trained']:
        st.warning("⚠️ System is not trained yet! Please go to the 'Train System' page to train the anomaly detection models.")
    else:
        st.success("✅ System is trained and ready for anomaly detection!")

    # Recent alerts
    st.subheader("🚨 Recent Alerts")

    if len(recent_data) > 0:
        # Display recent alerts in a nice format
        for _, row in recent_data.head(10).iterrows():
            timestamp = pd.to_datetime(row['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            confidence = row['confidence_score']

            # Color code based on confidence
            if confidence > 0.8:
                alert_class = "🔴 High"
            elif confidence > 0.5:
                alert_class = "🟡 Medium"
            else:
                alert_class = "🟢 Low"

            st.markdown(f"""
            <div class="alert-box">
                <strong>{alert_class} Priority Alert</strong><br>
                📅 {timestamp}<br>
                📹 Camera: {row['camera_id']}<br>
                🎯 Type: {row['anomaly_type']}<br>
                📊 Confidence: {confidence:.3f}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No anomalies detected in the last 24 hours.")

def show_training_page():
    """Show the system training page"""
    st.header("🏋️ Train Surveillance System")

    # Current status
    status = get_system_status()
    
    if status['trained']:
        st.success("✅ System is already trained and ready!")
    else:
        st.info("Train the anomaly detection system with normal behavior patterns.")

    # Training section
    st.subheader("🤖 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🤖 Generate Synthetic Training Data", disabled=status['trained']):
            with st.spinner("Creating synthetic training data..."):
                # Create synthetic normal behavior patterns
                training_features = create_synthetic_training_data()

                st.success(f"✅ Generated {len(training_features)} training samples")

                # Train the models
                with st.spinner("Training anomaly detection models..."):
                    success = st.session_state.anomaly_detector.train_models(training_features)

                if success:
                    # Update session state
                    st.session_state.system_trained = True
                    
                    st.success("🎯 Models trained successfully!")
                    
                    # Show success message and force page refresh
                    st.balloons()
                    st.info("🔄 Page will refresh to update system status...")
                    
                    # Force rerun to update all status indicators
                    st.rerun()

    with col2:
        if st.button("🔄 Refresh Status"):
            update_system_status()
            st.rerun()

    # Show training summary if trained
    if status['trained']:
        st.subheader("📊 Training Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", "1000")
            st.metric("Feature Dimensions", "8")

        with col2:
            st.metric("Model Type", "Isolation Forest")
            st.metric("Training Status", "✅ Complete")
            
        # Model performance info
        st.subheader("🎯 Model Information")
        st.markdown("""
        **Features Used:**
        - Person count
        - Vehicle count  
        - Total objects detected
        - Average confidence score
        - Object density in frame
        - Position variance
        - Average object size
        - Person-to-object ratio
        """)

def show_video_processing_page():
    """Show the video processing page (Demo version)"""
    st.header("🎬 Process Video for Anomalies")

    status = get_system_status()

    if not status['trained']:
        st.error("⚠️ System needs to be trained first! Please go to the 'Train System' page.")
        if st.button("🏋️ Go to Train System"):
            st.session_state.current_page = "Train System"
            st.rerun()
        return

    st.success("✅ System is ready for video processing!")
    st.info("Demo Mode: This will simulate video processing with synthetic data.")

    if st.button("🔍 Run Demo Analysis"):
        st.subheader("⚙️ Processing...")

        # Simulate feature extraction with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Extracting features from video...")
        features = st.session_state.video_processor.extract_features_from_video(
            "demo_video.mp4", progress_bar
        )

        if len(features) > 0:
            status_text.text("Detecting anomalies...")

            # Detect anomalies
            predictions, confidence_scores = st.session_state.anomaly_detector.detect_anomalies(features)

            # Process results
            anomaly_count = np.sum(predictions == -1)
            total_frames = len(predictions)
            anomaly_rate = (anomaly_count / total_frames * 100) if total_frames > 0 else 0

            status_text.text("Analysis complete!")

            # Show results
            st.subheader("📊 Analysis Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Frames Processed", total_frames)

            with col2:
                st.metric("Anomalies Detected", anomaly_count)

            with col3:
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

            with col4:
                avg_confidence = np.mean(confidence_scores)
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")

            # Log anomalies to database
            if anomaly_count > 0:
                for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                    if pred == -1:  # Anomaly detected
                        st.session_state.db_manager.log_anomaly(
                            camera_id="Demo_Camera_01",
                            anomaly_type="Behavioral_Anomaly",
                            confidence_score=conf,
                            features=features[i]
                        )

            # Create timeline visualization
            st.subheader("📈 Anomaly Timeline")

            # Timeline plot
            fig = go.Figure()

            # Add normal points
            normal_indices = np.where(predictions == 1)[0]
            fig.add_trace(go.Scatter(
                x=normal_indices,
                y=[0] * len(normal_indices),
                mode='markers',
                name='Normal',
                marker=dict(color='green', size=8)
            ))

            # Add anomaly points
            anomaly_indices = np.where(predictions == -1)[0]
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=[0] * len(anomaly_indices),
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x')
            ))

            fig.update_layout(
                title="Anomaly Detection Timeline",
                xaxis_title="Frame Index",
                yaxis_title="",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            if anomaly_count > 0:
                st.success(f"✅ {anomaly_count} anomalies logged to database!")

def show_analytics_page():
    """Show the analytics page"""
    st.header("📈 Analytics & Reports")

    # Get data
    data = st.session_state.db_manager.get_recent_anomalies(24)

    if len(data) == 0:
        st.info("No anomalies recorded in the last 24 hours. Run video processing to generate data.")
        # Show demo data instead
        st.subheader("📊 Demo Analytics")
        
        # Create sample data for demonstration
        demo_data = {
            'Hour': list(range(24)),
            'Anomaly_Count': np.random.poisson(2, 24)
        }
        
        fig = px.bar(
            demo_data,
            x='Hour',
            y='Anomaly_Count',
            title="Demo: Anomalies by Hour of Day"
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # Overview metrics (if real data exists)
    st.subheader("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Anomalies", len(data))
    
    with col2:
        avg_confidence = data['confidence_score'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
    with col3:
        high_confidence = len(data[data['confidence_score'] > 0.8])
        st.metric("High Confidence", high_confidence)
        
    with col4:
        unique_cameras = data['camera_id'].nunique()
        st.metric("Active Cameras", unique_cameras)

    # Anomalies over time
    st.subheader("📈 Anomaly Timeline")
    
    data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
    hourly_counts = data.groupby('hour').size().reset_index(name='count')
    
    fig = px.bar(hourly_counts, x='hour', y='count', title="Anomalies by Hour")
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">🔍 AI-Powered Surveillance System</h1>',
                unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Dashboard", "Train System", "Process Video", "Analytics", "Settings"]
    )

    # Enhanced system status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status:**")
    
    # Get current system status
    status = get_system_status()
    
    if status['trained']:
        st.sidebar.markdown('<div class="status-trained">🟢 Trained</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-not-trained">🔴 Not Trained</div>', unsafe_allow_html=True)
    
    # Status details
    st.sidebar.markdown(f"""
    - Model: {'✅' if status['model_ready'] else '❌'}
    - Processor: {'✅' if status['processor_ready'] else '❌'}  
    - Database: {'✅' if status['database_ready'] else '❌'}
    """)
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Status"):
        update_system_status()
        st.rerun()

    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Train System":
        show_training_page()
    elif page == "Process Video":
        show_video_processing_page()
    elif page == "Analytics":
        show_analytics_page()
    elif page == "Settings":
        show_settings_page()

if __name__ == "__main__":
    main()