package main

import (
	"net/http"
)

func main() {
	HttpHandler()
	http.ListenAndServe("127.0.0.1:7777", nil)
}
