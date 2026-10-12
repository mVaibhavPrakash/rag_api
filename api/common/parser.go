package common

import (
	"encoding/json"
	"fmt"
	"net/http"
)

func Marshal(data any) (string, error) {
	marshalData, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to encode patient payload: %w", err)
	} else {
		return string(marshalData), nil
	}
}

func UnMarshal(data []byte, p any) error {
	err := json.Unmarshal(data, &p)
	if err != nil {
		return fmt.Errorf("failed to decode patient payload: %w", err)
	}
	return nil
}

func GetHeader(r *http.Request) http.Header {
	return r.Header
}
