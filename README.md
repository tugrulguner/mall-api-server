# Mall API Server
This is the API server to receive data from edge devices and publish them to the [Dashboard Application](https://github.com/Build-with-AI-a-team/mall-covid-monitor)
This app is being made as part of #BuildwithAI hackathon.

### Architecture
- Frontend - Built with [Streamlit](https://www.streamlit.io/). It fetches the data from the Backend APIs. It also has a demo mode where it shows the fuctionality against randomly generated data.
- Backend - Built with Python [Flask](https://flask.palletsprojects.com/en/1.1.x/) and [TinyDB](https://tinydb.readthedocs.io/en/stable/). It hosts APIs to update and get data from the database.
- ML Model - Built with Python. It gets the data from the Cameras, and runs the algorithms to get the output. The Output is then sent to the Backend database using the APIs.
  
![Architecture Image](diagram.png)

## Notes
- The requirements.txt file contains the dependencies required for running the app in Heroku
- The models needs to be run separately using app\mainscript.py and app\yolo_main.py. Dependencies for the models are - 
    - tensorflow
    - opencv-contrib-python
    - numpy
    - imutils
    - setuptools==41.0.0

### Team Members
[Ignacio Amat](https://github.com/IgnacioAmat)  
[Dhruv Sheth](https://github.com/dhruvsheth-ai)  
[Tugrul Guner](https://github.com/tugrulguner)  
[Parijat Khan](https://github.com/Parijat29)  
[Ramona Rajagopalan](https://www.linkedin.com/in/ramona-rajagopalan/)  
[Arijit Das](https://github.com/arijitdas123student)  
[Deepak Vohra](https://www.linkedin.com/in/vohra-deepak/)