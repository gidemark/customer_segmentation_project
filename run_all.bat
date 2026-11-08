# run_all.bat - runs the full pipeline with Hadoop Windows binaries configured
@echo off
set HADOOP_HOME=%~dp0hadoop-bin
set PATH=%PATH%;%HADOOP_HOME%\bin
set PYSPARK_PYTHON=python
set PYSPARK_DRIVER_PYTHON=python
cd /d %~dp0
python src\01_data_cleaning.py
python src\02_feature_engineering.py
python src\03_clustering_model.py
python src\04_cluster_analysis.py