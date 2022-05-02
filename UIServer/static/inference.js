var started = false;

window.onload = function() {
    $("#start-button").click(function(event) {
        event.preventDefault();
        let lastRequestFinished = true;

        if (!started) {
            started = true;
            $.get("/open-hopper");
            $.get("/start-conveyor");
            $.get("/start-shaker");
            let id = setInterval(function() {
                // one request at a time
                if (lastRequestFinished) {
                    if (!started) {
                        clearInterval(id);
                        $.get("/close-hopper");
                        $.get("/stop-conveyor");
                        $.get("/stop-shaker");
                        $("body").css("background-color", "var(--darkGrey)");
                    } else {
                        lastRequestFinished = false;
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
                                started = false;
                                clearInterval(id);
                                $.get("/close-hopper");
                                $.get("/stop-conveyor");
                                $.get("/stop-shaker");
                                $("body").css("background-color", "green");
                            } else {
                                $("body").css("background-color", "var(--darkGrey)");
                            }

                            lastRequestFinished = true;
                        });
                    }
                }
            }, 500);
        }
    });

    $("#reset-button").click(function(event) {
        event.preventDefault();
        started = false;
        $("body").css("background-color", "var(--darkGrey)");
    });
}