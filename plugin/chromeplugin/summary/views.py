from django.http import HttpResponse, JsonResponse
from .funcs import summarize_url
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt


# Create your views here.


def index(request):
    return HttpResponse("I am a summarizer")


@csrf_exempt
@api_view(["POST"])
def summarize_web_page(request):
    url = request.data.get('url', '')

    # text = request.GET.get("text", "")
    summary = summarize_url(url)
    return JsonResponse({'result': summary})
