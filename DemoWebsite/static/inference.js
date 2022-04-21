var started = false;

window.onload = function() {
    $("#start-button").click(function(event) {
        event.preventDefault();

        if (!started) {
            setInterval(function() {
                $.get("/process-img", function(data, textStatus, jqXHR) {
                    let desiredClass = $("#desired-id").val();
                    let obj = JSON.parse(data);

                    $("#result-img").attr("src", obj.img);
                    let found = false;
                    for (let box of obj.boxes) {
                        if (box.class_name == desiredClass) {
                            found = true;
                        }
                    }

                    if (found) {
                        // handle found
                        $("body").css("background-color", "green");
                    } else {
                        $("body").css("background-color", "var(--darkGrey)");
                    }
                });
            }, 333);

            started = true;
        }
    });
}