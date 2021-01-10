# streamlit packages
import streamlit as st 
import os



def main():
	""" OCS Lemmatiser """
	
	# Description
	st.title("OCS Hybrid Lemmatiser")
	st.subheader("Demo")
	st.markdown("""
    	#### Description
    	This is an app for lemmatisation of OCS tokens by hybrid model
    	consisiting of seq2seq NN, dictionary, and linguistic rules 
    	""")
	
	#Lemmatisation
	token = st.text_area("Enter token to analyze...", "Type Here...")
	pos = st.text_area("Enter supposed PoS of token...", "Type Here...")
	
	st.write(token + pos)
	

	st.sidebar.subheader("About App")
	st.sidebar.text("OCS Lemmatiser")
	st.sidebar.info("Cudos to the Streamlit Team & Jesse E.Agbe (JCharis)")
	

if __name__ == '__main__':
	main()