#!/bin/bash

if [ -z "$1" ]; then
  echo "Pass test name or \"all\" to run all tests"
  exit
fi

if [ -z "$2" ]; then
  echo "Python path is not provided using default python3"
  TEST_NAME=$1
  PYTHON=python3
else
  PYTHON=$1
  TEST_NAME=$2
fi

if [ "$TEST_NAME" == "all" ]; then
  echo "Running all tests"
  $PYTHON -m unittest auto_tests.py
else
  echo "Running test $TEST_NAME"
  $PYTHON -m unittest auto_tests.$TEST_NAME
fi

