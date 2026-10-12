package patients

import (
	"net/http"
	"syncBitRAGAPI/api/patients/visits/draft"
)

func DraftHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "POST":
		draft.CreateDraft(w, r)
	case "PUT":
		draft.UpdateDraft(w, r)
	}
}

func VisitHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "POST":

	}
}
