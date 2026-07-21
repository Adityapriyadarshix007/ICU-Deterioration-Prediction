from setuptools import setup, find_packages

setup(
    name="icu-predictor-backend",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
        "pymongo==4.6.1",
        "python-dotenv==1.0.0",
        "pydantic==2.5.3",
        "bcrypt==4.1.2",
        "passlib==1.7.4",
        "python-jose[cryptography]==3.3.0",
    ],
)
