import streamlit as st

from src.ui.components import render_build_region_extractor

st.set_page_config(page_title="Build Region Extractor", page_icon="🧭", layout="wide")


def main():
    render_build_region_extractor()


if __name__ == "__main__":
    main()
