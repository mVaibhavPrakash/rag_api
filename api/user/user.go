package user

import (
	"io"
	"net/http"
	"syncBitRAGAPI/api/common"
)

func validateUserInfo(r *http.Request) {
	data, err := io.ReadAll(r.Body)
	if err != nil {
		panic("Unable to process request body.")
	}
	var user UserSchema

	error := common.UnMarshal(data, &user)
	if error != nil {
		panic("Unable to unmarshal the data. User data is corrupt.")
	}

	if user.FirstName == "" || user.LastName == "" || user.UserType == "" {
		panic("Incorrect user input provided")
	}

	CreateUser(user)
}

func CreateUser(user UserSchema) UserSchema {
	// DB handling

	return user
}
