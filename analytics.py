import datetime
import uuid

def log_event(event_name: str, **properties):
    """
    Appends an event to st.session_state["event_log"].
    If st is not available (e.g. running in tests), this silently passes or buffers.
    """
    try:
        import streamlit as st
    except ImportError:
        return  # Silently skip if streamlit is not installed (e.g. in some isolated test env)
        
    try:
        if "event_log" not in st.session_state:
            st.session_state["event_log"] = []
            
        event = {
            "event_name": event_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **properties
        }
        st.session_state["event_log"].insert(0, event) # prepend so newest is first
    except Exception:
        # Outside of a streamlit script run context, st.session_state will raise an error.
        # We catch and ignore it so tests don't crash when testing retrieval logic directly.
        pass
