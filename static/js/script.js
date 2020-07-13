$(function () {
    playButton = $('#play-button')

    var dir_igos;
    var dir_gcam;

    var ins_curve;
    var ins_curve_gcam;
    var del_curve;
    var del_curve_gcam;

    var data_ins;
    var data_del;
    var data_ins_gcam;
    var data_del_gcam;

    var range
        // define the layout
    var layout1 = {
      autosize: false,
      width: 250,
      height: 250,
      margin: {
        l: 40,
        r: 10,
        b: 30,
        t: 25,
        pad: 4
      },
      xaxis: {
        range: [-0.05, 1.01],
        fixedrange: true,
        autotick: false,
        ticks: 'outside',
        tick0: 0,
        dtick: 0.2,
        ticklen: 2,
        tickwidth: 1,
        tickcolor: '#000'
      },
      yaxis: {
        range: [-0.05, 1.01],
        fixedrange: true,
        autotick: false,
        ticks: 'outside',
        tick0: 0,
        dtick: 0.1,
        ticklen: 0,
        tickwidth: 0,
        tickcolor: '#000'
      },
      annotations: [{
        x: 0.5,
        y: 0.5,
        xref: 'x',
        yref: 'y',
        text: '',
        showarrow: false,
        ax: 0,
        ay: 0
      }],
    };

    var layout2 = jQuery.extend({}, layout1)
    var layout3 = jQuery.extend({}, layout1)
    var layout4 = jQuery.extend({}, layout1)
    var config = {responsive: true}

    // functions
    function range_num(start, end)
    {
        var array = new Array();
        for(var i = start; i < end; i++)
        {
            array.push(i/(end-1));
        }
        return array;
    }

    function auc(arr) {
      return (arr.reduce((x, y) => x+y, 0) - arr[0] / 2 - arr[arr.length-1] / 2) / 196;
    };

    function update_data(data, raw_data, layout, m, name, color){
        // var m = m/4 ;

        data.y = raw_data.slice(0,m)
        data.x = [...Array(m).keys()].map(x => x / 194);
        // var ins = data.slice(0,m)
        // // var del = del_curve.slice(0,m);
        // var rng = [...Array(m).keys()].map(x => x / 48);

        if (m > 1){
            layout.annotations[0].text = 'AUC: ' + auc(data.y).toFixed(3)
        } else {
            layout.annotations[0].text = ' '
        }

        return [data]
    }

    function plotly_data(xdata, ydata, namedata, color='#35677B'){
        var trace = {
          x: xdata,
          y: ydata,
          fill: 'tozeroy',
          type: 'scatter',
          mode: 'line',
          // name: namedata,
          line: {color: color}
        };
        return trace;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    $('#run').click(function () {
        var input_url = $('#url').val();

         $('source').removeAttr("src")
        // hidden video section
        $("#my-video").attr("hidden", 'hidden')

        $("#image_holder").removeAttr('hidden')

        //hidden saliency
        $("#igos-holder").attr("hidden", 'hidden')
        $("#gcam-holder").attr("hidden", 'hidden')
        $("#ins-del-holder").attr("hidden", 'hidden')
        $("#ins-del-holder-gcam").attr("hidden", 'hidden')

        // reset plotl
        Plotly.purge('line-chart-ins')
        Plotly.purge('line-chart-del')
        Plotly.purge('line-chart-ins-gcam')
        Plotly.purge('line-chart-del-gcam')

        // reset text adn slider
        $("#cat-name").text("")
        $("#slider").attr("hidden", 'hidden')
        $("#slide_note").attr("hidden", 'hidden')
        $("#play-button").attr("hidden", 'hidden')

        if (input_url.length > 0) {
            //show results section
            $('#results').removeAttr('hidden')

            //show image
            $("#selected-image").attr("src", input_url);
            $("#image_holder").removeAttr('hidden')

            $.ajax({
                url: '/metricIGOS',
                data: {
                    "url": input_url
                },
                method: 'POST',
                success: function (response) {
                    // hide original image
//                    $("#image_holder").attr("hidden", 'hidden')
                    dir_igos = response["out_dir"]
                    dir_gcam = response["gcam_dir"]

                    // show saliency
                    $("#igos-image").attr("src", dir_igos + '_cam.png')
                    $("#gcam-image").attr("src", dir_gcam + '_cam.png')
                    $("#cat-name").text('Explanation for : {' + response['cat_name'] + '}')
                    $("#igos-holder").removeAttr('hidden')
                    $("#gcam-holder").removeAttr('hidden')

                    // show insertion deletion initial images
                    $("#ins-image").attr("src", dir_igos + 'ins_0.jpg')
                    $("#ins-image-gcam").attr("src", dir_gcam + 'ins_0.jpg')
                    $("#del-image").attr("src", dir_igos + 'del_0.jpg')
                    $("#del-image-gcam").attr("src", dir_gcam + 'del_0.jpg')
                    $("#ins-del-holder").removeAttr('hidden')
                    $("#ins-del-holder-gcam").removeAttr('hidden')

                    // get ins del curves to plot
                    ins_curve = response['ins_c'].split(',').map(x=>+x)
                    ins_curve_gcam = response['ins_c_gcam'].split(',').map(x=>+x)
                    del_curve = response['del_c'].split(',').map(x=>+x)
                    del_curve_gcam = response['del_c_gcam'].split(',').map(x=>+x)

                    range = range_num(1, ins_curve.length + 1);

                    data_ins = plotly_data(range, ins_curve, 'insertion', '#35677B');
                    data_del = plotly_data(range, del_curve, 'deletion', '#973253');
                    data_ins_gcam = plotly_data(range, ins_curve_gcam, 'insertion', '#35677B');
                    data_del_gcam = plotly_data(range, del_curve_gcam, 'deletion', '#973253');

                    layout1.annotations[0].text = 'AUC: ' + auc(ins_curve).toFixed(3)
                    // plot all -- initial
                    Plotly.plot(`line-chart-ins`, [data_ins], layout1, config);

                    // layout1.title = 'Insertion'
                    layout2.annotations[0].text = 'AUC: ' + auc(del_curve).toFixed(3)
                    Plotly.plot(`line-chart-del`, [data_del], layout2, config);

                    layout3.annotations[0].text = 'AUC: ' + auc(ins_curve_gcam).toFixed(3)
                    Plotly.plot(`line-chart-ins-gcam`, [data_ins_gcam], layout3, config);

                    layout4.annotations[0].text = 'AUC: ' + auc(del_curve_gcam).toFixed(3)
                    Plotly.plot(`line-chart-del-gcam`, [data_del_gcam], layout4, config);
                    //slider

                    $("#slider").removeAttr('hidden')
                    $("#slide_note").removeAttr("hidden")
                    $("#play-button").removeAttr("hidden")
                    $("#slider").slider('value', 0);

                },
                error: function (error) {
                    console.log(error);
                }
            });
        }
    })



    // load images to the carousel
    var image_numbers = 12
    var images_in_carousel = 6
    var images_rounded = image_numbers - image_numbers % images_in_carousel

    // carousel items
    for (var i = 1; i <= (images_rounded / images_in_carousel); i++) {
        if (i == 1) {
            $('.carousel-inner').append('<div class="carousel-item active" id="item' + i + '"></div>')
        } else {
            $('.carousel-inner').append('<div class="carousel-item" id="item' + i + '"></div>')
        }
    }

    // filling the items with images
    for (var i = 0; i < (images_rounded / images_in_carousel); i++) {
        for (var j = 1; j <= images_in_carousel; j++) {
            $("#item" + (i + 1)).append(
                '<div class="col-md-2 mb-3"><div class="card"> <a href="#results"><img class="" src="/static/carousel/images/' + (i * images_in_carousel + j) + '.jpg" alt="Card image cap" height="150" width="150" value="' + (i * images_in_carousel + j) + '"></a></div></div>'
            )
        }
    }

        // send Ajax after click on image
    $('#carousel .carousel-inner img').click(function () {
        var image_selected = $(this).attr('value')
        var input_url = '$%#carousel' + image_selected

        $('source').removeAttr("src")
        // hidden video section
        $("#my-video").attr("hidden", 'hidden')

        $("#image_holder").removeAttr('hidden')

        //hidden saliency
        $("#igos-holder").attr("hidden", 'hidden')
        $("#gcam-holder").attr("hidden", 'hidden')
        $("#ins-del-holder").attr("hidden", 'hidden')
        $("#ins-del-holder-gcam").attr("hidden", 'hidden')

        // reset plotl
        Plotly.purge('line-chart-ins')
        Plotly.purge('line-chart-del')
        Plotly.purge('line-chart-ins-gcam')
        Plotly.purge('line-chart-del-gcam')

        // reset text adn slider
        $("#cat-name").text("")
        $("#slider").attr("hidden", 'hidden')
        $("#slide_note").attr("hidden", 'hidden')
        $("#play-button").attr("hidden", 'hidden')

        if (input_url.length > 0) {
            //show results section
            $('#results').removeAttr('hidden')

            //show image
            $("#selected-image").attr("src", "/static/carousel/images/"+image_selected+'.jpg');
            $("#image_holder").removeAttr('hidden')

            $.ajax({
                url: '/metricIGOS',
                data: {
                    "url": input_url
                },
                method: 'POST',
                success: function (response) {
                    // hide original image
//                    $("#image_holder").attr("hidden", 'hidden')
                    dir_igos = response["out_dir"]
                    dir_gcam = response["gcam_dir"]
                    // show saliency
                    $("#igos-image").attr("src", dir_igos + '_cam.png')
                    $("#gcam-image").attr("src", dir_gcam + '_cam.png')
                    $("#cat-name").text('Explanation for : {' + response['cat_name'] + '}')
                    $("#igos-holder").removeAttr('hidden')
                    $("#gcam-holder").removeAttr('hidden')

                    // show insertion deletion initial images
                    $("#ins-image").attr("src", dir_igos + 'ins_0.jpg')
                    $("#ins-image-gcam").attr("src", dir_gcam + 'ins_0.jpg')
                    $("#del-image").attr("src", dir_igos + 'del_0.jpg')
                    $("#del-image-gcam").attr("src", dir_gcam + 'del_0.jpg')
                    $("#ins-del-holder").removeAttr('hidden')
                    $("#ins-del-holder-gcam").removeAttr('hidden')

                    // get ins del curves to plot
                    ins_curve = response['ins_c'].split(',').map(x=>+x)
                    ins_curve_gcam = response['ins_c_gcam'].split(',').map(x=>+x)
                    del_curve = response['del_c'].split(',').map(x=>+x)
                    del_curve_gcam = response['del_c_gcam'].split(',').map(x=>+x)

                    range = range_num(1, ins_curve.length + 1);

                    data_ins = plotly_data(range, ins_curve, 'insertion', '#35677B');
                    data_del = plotly_data(range, del_curve, 'deletion', '#973253');
                    data_ins_gcam = plotly_data(range, ins_curve_gcam, 'insertion', '#35677B');
                    data_del_gcam = plotly_data(range, del_curve_gcam, 'deletion', '#973253');

                    layout1.annotations[0].text = 'AUC: ' + auc(ins_curve).toFixed(3)
                    // plot all -- initial
                    Plotly.plot(`line-chart-ins`, [data_ins], layout1, config);

                    // layout1.title = 'Insertion'
                    layout2.annotations[0].text = 'AUC: ' + auc(del_curve).toFixed(3)
                    Plotly.plot(`line-chart-del`, [data_del], layout2, config);

                    layout3.annotations[0].text = 'AUC: ' + auc(ins_curve_gcam).toFixed(3)
                    Plotly.plot(`line-chart-ins-gcam`, [data_ins_gcam], layout3, config);

                    layout4.annotations[0].text = 'AUC: ' + auc(del_curve_gcam).toFixed(3)
                    Plotly.plot(`line-chart-del-gcam`, [data_del_gcam], layout4, config);
                    //slider

                    $("#slider").removeAttr('hidden')
                    $("#slide_note").removeAttr("hidden")
                    $("#play-button").removeAttr("hidden")
                    $("#slider").slider('value', 0);

                },
                error: function (error) {
                    console.log(error);
                }
            });
        }
    })

    // var moving = false;
    playButton.on("click", function() {
         var button = $(this);
         if (button.text() == "Pause" ){
            // moving = false;
            clearInterval(timer)
            button.text("Play")
         }
             else
         {
             button.text("Pause")
             // moving = true;
             timer = setInterval(step, 350);
         }

    })


    function step() {
        var val = $('#slider').slider("option", "value");
        $("#slider").slider('value', val+1);
    }

    var $loading = $('#loading').hide();
    $(document)
        .ajaxStart(function () {
            $loading.show();
        })
        .ajaxStop(function () {
            $loading.hide();
        });


    $("#slider").slider({
        value:  0,
        min:    0,
        max:    196,
        step:   1,
        animate: "fast",
        slide: function(event, ui){
            update_comps(ui.value)
        },
        change: function(event, ui){
            update_comps(ui.value)

            if (ui.value == 196) {
                playButton.text("Play")
                clearInterval(timer)
            }
        },
    });

    function update_comps(val) {
        Plotly.update(`line-chart-ins`,
                      update_data(data_ins, ins_curve, layout1, val),
                      layout1,
        )
        Plotly.update(`line-chart-del`,
                      update_data(data_del, del_curve, layout2, val),
                      layout2,
        )
        Plotly.update(`line-chart-ins-gcam`,
                      update_data(data_ins_gcam, ins_curve_gcam, layout3, val),
                      layout3,
        )
        Plotly.update(`line-chart-del-gcam`,
                      update_data(data_del_gcam, del_curve_gcam, layout4, val),
                      layout4,
        )

        $("#ins-image").attr("src", dir_igos  + 'ins_'+val+'.jpg')
        $("#del-image").attr("src", dir_igos  + 'del_'+val+'.jpg')

        $("#ins-image-gcam").attr("src", dir_gcam  + 'ins_'+val+'.jpg')
        $("#del-image-gcam").attr("src", dir_gcam  + 'del_'+val+'.jpg')
    }
    // if a new query is arrvied, ignore the previous ones


    // if select image from carousel, return the video

	    $("#igos-image").hover(
        function () {
            $(this).attr("src", dir_igos + '_heatmap.png')
        },
        function () {
            $(this).attr("src", dir_igos + '_cam.png')
        }
    )
    $("#gcam-image").hover(
        function () {
            $(this).attr("src", dir_gcam + '_heatmap.png')
        },
        function () {
            $(this).attr("src", dir_gcam + '_cam.png')
        }
    )


});


