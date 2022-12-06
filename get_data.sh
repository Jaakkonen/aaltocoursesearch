#!/bin/bash

if [[ -z "$COURSE_API_TOKEN" ]]; then
    echo "Please set COURSE_API_TOKEN to one obtained from 3scale.ocp.aalto.fi SISU api."
    exit
fi
curl "https://course.api.aalto.fi/api/sisu/v1/courseunitrealisations?USER_KEY=$COURSE_API_TOKEN" > courseunitrealizations.json
