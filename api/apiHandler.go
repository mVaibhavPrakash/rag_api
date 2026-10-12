package main

import (
	"net/http"
	"syncBitRAGAPI/api/patients"
)

func HttpHandler() {
	http.HandleFunc("/patients/{patientId}/encounters/{encountersId}/draft", patients.DraftHandler)
}
