"""
Funciones auxiliares de autenticación para el dashboard de portfolio.

La autenticación es deliberadamente simple porque esta aplicación es una
analytical demo: no usa variables de entorno, secrets ni base de datos.
"""

import hmac

import streamlit as st

DEMO_USERNAME = "demo"
DEMO_PASSWORD = "portfolio"


def autenticar_usuario(username: str, password: str) -> bool:
    """Valida las credenciales del usuario demo."""
    username_valido = hmac.compare_digest(str(username).strip(), DEMO_USERNAME)
    password_valido = hmac.compare_digest(str(password).strip(), DEMO_PASSWORD)
    return username_valido and password_valido


def login() -> None:
    """Renderiza la pantalla de ingreso."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if "authenticated_user" not in st.session_state:
        st.session_state["authenticated_user"] = None

    st.markdown(
        """
        <style>
        div[data-testid="stAppViewContainer"] {
            background: #f7f9fb;
        }

        div[data-testid="stVerticalBlock"]:has(#mic-login-marker) {
            max-width: 460px;
            margin: 120px auto 0 auto;
            padding: 36px 34px 32px 34px;
            background: rgba(255, 255, 255, 0.94);
            border-radius: 18px;
            box-shadow: 0 16px 38px rgba(0,0,0,0.16);
            box-sizing: border-box;
            overflow: hidden;
        }

        div[data-testid="stVerticalBlock"]:has(#mic-login-marker) * {
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        .mic-login-title {
            text-align: center;
            font-size: 2.0rem;
            font-weight: 900;
            letter-spacing: 0.2px;
            margin: 0 0 10px 0;
        }

        .mic-login-desc {
            text-align: center;
            font-size: 1.0rem;
            line-height: 1.55;
            color: #6b7280;
            margin: 0 0 22px 0;
        }

        .mic-login-separator {
            width: 80px;
            height: 4px;
            margin: 0 auto 26px auto;
            background-color: #2BB3A3;
            border-radius: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<span id="mic-login-marker"></span>', unsafe_allow_html=True)
    st.markdown('<div class="mic-login-title">Customer Value and Retention</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mic-login-desc">Analytical demo for exploring segmentation, churn risk and customer lifetime value.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="mic-login-separator"></div>', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")

    if submitted:
        if autenticar_usuario(username, password):
            st.session_state["logged_in"] = True
            st.session_state["authenticated_user"] = username
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")


def logout() -> None:
    """Cierra la sesión actual."""
    st.session_state["logged_in"] = False
    st.session_state["authenticated_user"] = None


def proteger_dashboard() -> None:
    """Detiene la app si el usuario no inició sesión."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if "authenticated_user" not in st.session_state:
        st.session_state["authenticated_user"] = None

    if not st.session_state["logged_in"]:
        login()
        st.stop()
