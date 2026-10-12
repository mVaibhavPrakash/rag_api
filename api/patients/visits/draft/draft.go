package draft

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"syncBitRAGAPI/api/common"
	"time"
	"uuid"
)

type FrequencyUnit string
type DosageUnit string
type DraftStatus string

const (
	Hour FrequencyUnit = "Hour"
	OD   FrequencyUnit = "Day Time"
	ON   FrequencyUnit = "Night Time"
	DN   FrequencyUnit = "Both Day and Night"
	TH   FrequencyUnit = "Thrice a day" // Day, Afternoon and Night

	Tabs       DosageUnit = "Tablets"
	Ml         DosageUnit = "Milliliter"
	TableSpoon            = "Tablespoon"

	Drafted   DraftStatus = "Drafted"
	Completed DraftStatus = "Completed"
)

type MedicationItem struct {
	Name                string        `json:"name"`
	Duration            string        `json:"duration"` // How many days
	Dosage              DosageUnit    `json:"dosage"`   // Number of tabs/ml/tbs at a time
	DosageFrequencyUnit FrequencyUnit `json:"dosageFrequency"`
}

type FirstDraft struct {
	Medication []MedicationItem `json:"medication"`
}

type Draft struct {
	Id string `json:"id"`
	FirstDraft
	Version   int    `json:"version"`
	Status    string `json:"status"`
	createdAt time.Time
	updatedAt time.Time
}

func (p *FirstDraft) Validate() bool {
	if len(p.Medication) != 0 {
		return true
	}
	return false
}

func (p *Draft) Validate() bool {
	if p.Id != "" && p.Version != 0 && p.Status != "" {
		return true
	}
	return false
}

func createNewDraft(p *FirstDraft) Draft {
	var draft Draft
	draft.Id = uuid.NewV4().String()
	draft.FirstDraft = *p
	draft.Status = string(Drafted)
	draft.Version = 1

	return draft
}

func UpdateDraft(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			fmt.Fprintf(w, "%v", rec)
			w.WriteHeader(http.StatusInternalServerError)
		}
		r.Body.Close()
	}()

	responseBody, err := io.ReadAll(r.Body)

	if err != nil {
		panic(err)
	}

	var draft Draft

	unMarshalError := common.UnMarshal(responseBody, &draft)

	if unMarshalError != nil {
		panic(unMarshalError)
	}

	if draft.Validate() {
		// Verify if data is stale and then accordingly save the data

		marshalData, _ := common.Marshal(draft)
		w.Write([]byte(marshalData))
	}
}

func CreateDraft(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			fmt.Fprintf(w, "Something wrong! Try again after sometime %v", rec)
			w.WriteHeader(http.StatusInternalServerError)
		}
		r.Body.Close()
	}()
	header := common.GetHeader(r)
	bearerToken := header.Get("Bearer")

	if bearerToken == "" {
		http.Error(w, "Unauthorized access, Bearer token in not present", http.StatusUnauthorized)
		return
	}

	data, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("read request body: %v", err)
		http.Error(w, "Unable to read request body", http.StatusBadRequest)
		return
	}

	var fDraft FirstDraft
	unmarshalError := common.UnMarshal(data, &fDraft)

	if unmarshalError != nil || !fDraft.Validate() {
		panic(fmt.Sprintf("Error unmarshal data. Please try again with correct data input. %v", unmarshalError))
	}

	draftedData := createNewDraft(&fDraft)

	// Save to db

	d, error := common.Marshal(draftedData)

	if error != nil {
		panic(fmt.Sprintf("Error unmarshal data. %v", draftedData))
	}

	w.Write([]byte(d))
}
