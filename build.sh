#!/bin/bash

jupytext --to notebook clustering.py
jupyter nbconvert --to html --execute clustering