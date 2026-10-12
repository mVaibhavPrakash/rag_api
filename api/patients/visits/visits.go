package visits

type VisitStatus string

const (
	Scheduled  VisitStatus = "Scheduled"
	InProgress VisitStatus = "In-Progress"
	Completed  VisitStatus = "Completed"
)

type Visit struct {
	id        string
	patientId string
	doctorId  string
	createdAt string
	startAt   string
	endAt     string
	note      string // Like next visit scheduled after 10 days etc
	status    VisitStatus
}
