# setup_env.bat - configures Hadoop Windows binaries for Spark on Windows
@echo off
set HADOOP_HOME=%~dp0hadoop-bin
set PATH=%PATH%;%HADOOP_HOME%\bin
echo HADOOP_HOME=%HADOOP_HOME%