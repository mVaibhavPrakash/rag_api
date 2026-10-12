package user

import "time"

type UserType string

const (
	UserTypeDoctor  UserType = "Doctor"
	UserTypePatient UserType = "Patient"
	UserTypeStaff   UserType = "Staff"
	UserTypeClinic  UserType = "Clinic"
)

type UserSchema struct {
	Id         string   `json:"id"`
	FirstName  string   `json:"firstName"`
	MiddleName string   `json:"middleName"`
	LastName   string   `json:"lastName"`
	Age        int      `json:"age"`
	UserType   UserType `json:"userType"`
	createdAt  time.Time
	updatedAt  time.Time
}

type AddressSchema struct {
	ownerId     string // Can be patient, staff, clinic, doctor etc
	OwnerType   string `json:"ownerType"` // Used to identify if address if for a clinic, patient, staff, doctor etc
	id          string
	AddressLine string `json:"addressLine"`
	City        string `json:"city"`
	State       string `json:"state"`
	Country     string `json:"country"`
	updatedBy   string // Id of the user who updated this record
	createdAt   time.Time
	updatedAt   time.Time
}
