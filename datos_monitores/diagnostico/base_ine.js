$(document).ready(function(){
	menuActivo = false;
	$(".elmNivel1.abrir").click(function(){
		visible = $(this).next(".subNivel").toggle().is(":visible");
		if(visible){
			$(this).addClass("elmActivo");
		}else{
			$(this).removeClass("elmActivo");
		};
	});
	$(".elmNivel2.abrir,.abrirSinHijos").click(function(){
		visible = $(this).addClass("elmActivo").next("ul").toggle().is(":visible");
		if(visible){
			$(this).addClass("elmActivo");
		}else{
			$(this).removeClass("elmActivo");
		};
	});
	$(".menuHeader").mouseenter(function(){
		elm = $(this);
		setTimeout(function(){
			if( menuActivo && elm.children().hasClass("menuEscritorioAbierto") ){
				elm.addClass("menuHeaderActivo");
				elm.children(".menuEscritorioAbierto").show();
			}
		}, 300);
		menuActivo = true;
	});
	$(".menuHeader").mouseleave(function(){
		$(this).removeClass("menuHeaderActivo");
		$(".menuEscritorioAbierto").hide();
		menuActivo = false;
	}).mouseleave();

	$(".menuMovilIcono").click(function(){
		$(".menuMovil").animate({ left:"0" },500);
		$(".menuMovilOverlay").fadeIn('slow');
	});
	$(".contCerrarMovil i").click(function(){
		$(".menuMovilOverlay").fadeOut('slow');
		$(".menuMovil").animate({ left:"-300px" },500);	
	});
	
	modTxtXL  = false;
	modTxtXS  = false;

	$(".opcAccesibilidad").click(function(){
			$(".opcAccesibilidad").removeClass("accesibilidadActivo");
			sel = $(this);
			if(sel.attr("id") == "textoGrande"){
				if(!modTxtXL){
					$("html").css("font-size","19px");
					modTxtXL = true;
					modTxtXS = false;
					sel.addClass("accesibilidadActivo");
				}else{
					$("html").css("font-size","inherit");
					modTxtXL = false;
				}
			}
			else if(sel.attr("id") == "textoChico"){
				if(!modTxtXS){
					$("html").css("font-size","13px");
					modTxtXS = true;
					modTxtXL = false;
					sel.addClass("accesibilidadActivo");
				}else{
					$("html").css("font-size","inherit");
					modTxtXS = false;
				}
			}	
	});

	$(".opcAccesibilidadBorrar").click(function(){
		$("html").removeAttr("style");
		$(".opcAccesibilidad,#textoByN").removeClass("accesibilidadActivo");
		modTxtXS = false;
		modTxtXL = false;
	});
	gris = false;
	$("#textoByN").click(function(){	
		if(!gris){
			$("html").css("filter","grayscale(100%)");
			gris = true;
			$(this).addClass("accesibilidadActivo");
		}else{
			$("html").css("filter","unset");
			gris = false;
			$(this).removeClass("accesibilidadActivo");
		}
	});

	$(".accesibilidadAbierto .fa-times-circle").click(function(){
		$(".accesibilidadAbierto").animate({right:"-275px"},500);
	});

	$(".accesibilidad").click(function(){
		$(".accesibilidadAbierto").animate({right:"0"},500);
	});
	
	$(".verIndicadoresPrincipalesBtn").click(function(){
		$(".indicadoresInteriores").toggle();
	});
	
	
	
	$(".busquedaAbierta .fa-times-circle").click(function(){
		$(".busquedaAbierta").hide();
	});
	
	
	
	
	$(".busqueda").click(function(){
		$(".busquedaAbierta").show();
		$(".busquedaAbierta .k-input").attr("placeholder","Buscar");
		$(".busquedaAbierta .k-input").focus();
	});
	

});

/*
function buscar(t){
	window.location = '/player_detail?username=' + t;
}
*/







