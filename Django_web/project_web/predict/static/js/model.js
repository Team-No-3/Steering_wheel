//model.js는 html이 받은 응답을 보낸다
//js는 주석이 //

//document.write는 괄호안을 출력한다
//아래는 git에 있는 이미지 리사이즈를 의미한다
document.write('<script src="https://cdn.jsdelivr.net/gh/ericnograles/browser-image-resizer@2.4.0/dist/index.js"></script>');
const resize_config = {
    quality: 1.0,
    maxWidth: 512,
    maxHeight: 512
};

//document.getElementById. ID에 접근하기 위한것 js
//constant로 변함없는 값을 설정함 js
const fileInput = document.getElementById('file');
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const preview = document.getElementById("preview");
const original = document.getElementById("original");

print_index = 0


// timer1은 타이머설정을 위한 것으로 동작시행을 위한 시간을 의미
var timer1 = null;

//ajax: javascript의 페이지 일부를 고쳐서 로드하는 방식(비동기 통신), 불필요리소스 낭비를 방지
//async는 javascript, 프라미스를 반환, 프라미스란 서버에서 받은 데이터를 화면에 표시할 때 사용
//var 함수선언
//csrftoken 매요청 시 임의 난수생성 저장 - 보안의 목적
//getcookiejs 쿠키를 저장한다, 쿠키란 페이지의 저장소
//append headers사이에 요소들을 삽입 -> header/'X-CSRFToken'/csrftoken

//ajax로 요철을 보낼때는 header에 x-csrftoken이라는 키를 value로 지정하면 된다. = 보안
//아래는 이미지 리사이즈에 해당하는 내용이다 git에 양식이 정해져 있다
const predict = async () => {
    var headers = new Headers();
    var csrftoken = getCookieJS('csrftoken');
    headers.append('X-CSRFToken', csrftoken);

    //fileinput 파일수정이 쉽다
    const files = fileInput.files;

    //[]map()은 []의 요소에 ()를 모두 적용한다
    //[...files]는 시작점부터 끝까지
    //let 변수재할당가능, const 불가능, var는 중복선언가능(마지막것으로), let과 const는 불가능
    //await는 async안에서 작동하며 프라미스 처리를 기다린다.
    //formdata는 html을 쉽게 전송하는 객체이다
    //fetch는 첫인자로 url 두번째로 옵션 객체를 받아 프로미스타입의 객체를 반환한다, 함수가 정해져있다
    [...files].map(async (img) => {
        let resizedImage = await BrowserImageResizer.readAndCompressImage(img, resize_config);
        const data = new FormData();
        data.append('file', resizedImage);
        const result = await fetch("/api/predict/",
            {
                method: 'POST',
                headers: headers,
                credentials: 'include',
                body: data,
            }).then(response => {
                return response.blob();            
            }).catch((error) => {
                return 'ERROR';
            });
            
        renderImage(result, preview);
        
        //이미지를 가져오고 시간 후에 버튼 클리어 누른다
        if (timer1 != null){
            clearInterval(timer1);
        }
        timer1 = setTimeout(() => clearButton.click(), 1000);
    })
};


//blob는 이미지같은 멀티미디어 데이터를 다룰 때 사용, 데이터 사이즈와 타입을 받는다
//filereader는 버퍼를 읽는 
//readasdataurl은 특정 blob을 읽어온다
//.onloadend는 로딩하는 것으로 ()의 조건에 {}를 작동시킨다 -> 무조건 이
const renderImage = (imageblob, print_row) => {
    const reader = new FileReader();
    const blob = imageblob // blob or file
    reader.readAsDataURL(blob); 
    reader.onloadend = () => {
        const base64data = reader.result;

        // innerhtml은 html요소에 접근하여 변경한다
        print_row.innerHTML = `
          <img src="${base64data}" style="max-width: 250px; height: auto;">`;

    };
};

//getcookiejs는 도메인 변환을 위한 것으로 우리는 a레코드 보다 cname이 적합하다
//cname은  도메인 네임, 하나의 ip주소에서 여러 개 서비스를 실행할때(ip주소가 자주 바뀔때) 유연한 대응가능
//decodeURIComponent는 문자열을 디코딩한다, 주소를 읽기 쉽게하기 위함 -> 주소가 name=안녕하세요, 이스케이핑
//split 은 구분자를 이용하여 문자열로 나눈다

const getCookieJS = (cname) => {
    var name = cname + "=";
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for(var i = 0; i <ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "";
}


//addEventListener는 이벤트등록 방식으로 여러 이벤트 핸들러를 등록할 수 있다 change는 값이 변경되었을 때 실행한다
//이미지를 새로 받는다
fileInput.addEventListener("change", () => {
    const reader = new FileReader();
    const blob = fileInput.files[0];
    reader.readAsDataURL(blob); 
    reader.onloadend = () => {
        original.innerHTML = `<img src="${reader.result}" style="max-width: 250px; height: auto; margin: 100px;">`;
    }
    //버그방지용
    if (timer1 != null){
        clearInterval(timer1);
    }
    //시행, 파일을 고르면, 0.5s 뒤 예측버튼을 누른다
    timer1 = setTimeout(() => predictButton.click(), 500);
});



predictButton.addEventListener("click", () => predict());


//clear는 본래 이미지, 모델 이미지, 파일 이름을 지운다
clearButton.addEventListener("click", () => {
    original.innerHTML=""; 
    preview.innerHTML = "";
    fileInput.value = "";
}  );