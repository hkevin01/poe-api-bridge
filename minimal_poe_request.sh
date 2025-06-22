#!/bin/bash

# ABSOLUTE MINIMAL POE API REQUEST
# Only 2 headers required: content-type and poegraphql

curl 'https://poe.com/api/gql_POST' \
  -H 'content-type: application/json' \
  -H 'poegraphql: 1' \
  --data-raw '{"queryName":"ExploreBotsListPaginationQuery","variables":{"categoryName":"defaultCategory","count":20,"cursor":null},"extensions":{"hash":"b24b2f2f6da147b3345eec1a433ed17b6e1332df97dea47622868f41078a40cc"}}'