// Fetches the OVMS VAL Jenkins Pipelines wiki page and lists daily recompile jobs (group_2)

pipeline {
    agent any
    parameters {
        string(name: 'WIKI_PAGE_ID',
               defaultValue: '3413673559',
               description: 'Confluence page ID for OVMS VAL Jenkins Pipelines')
    }
    stages {
        stage('List daily recompile jobs (group_2)') {
            steps {
                script {
                    def wikiBaseUrl = 'https://wiki.ith.intel.com'
                    def apiUrl = "${wikiBaseUrl}/rest/api/content/${params.WIKI_PAGE_ID}?expand=body.storage"

                    echo "Fetching wiki page: ${wikiBaseUrl}/spaces/OVMS/pages/${params.WIKI_PAGE_ID}"

                    withCredentials([usernamePassword(
                        credentialsId: 'confluence-credentials',
                        usernameVariable: 'WIKI_USER',
                        passwordVariable: 'WIKI_PASS')]) {

                        def response = httpRequest(
                            url: apiUrl,
                            customHeaders: [[name: 'Authorization',
                                             value: "Basic ${"${WIKI_USER}:${WIKI_PASS}".bytes.encodeBase64()}",
                                             maskValue: true]],
                            acceptType: 'APPLICATION_JSON',
                            httpMode: 'GET'
                        )

                        def json = readJSON text: response.content
                        def html = json.body.storage.value

                        // Extract all Jenkins job URLs from the page
                        def allUrls = (html =~ /https:\/\/ci\.iotg\.sclab\.intel\.com\/job\/[^"<\s]+/).collect { it.replaceAll(/\/+$/, '') }
                        allUrls = allUrls.unique()

                        // The wiki page lists URLs in order. The "daily recompile jobs (group_2)" section
                        // starts after "daily recompile jobs" heading and ends at "weekly recompile jobs" heading.
                        // We identify these by the URL order matching the page structure.
                        //
                        // Strategy: find the section markers in HTML and extract URLs between them.
                        def lowerHtml = html.toLowerCase()
                        def startIdx = lowerHtml.indexOf('daily recompile jobs')
                        def endIdx = lowerHtml.indexOf('weekly recompile jobs')

                        if (startIdx == -1) {
                            error "Could not find 'daily recompile jobs' section on the wiki page"
                        }
                        if (endIdx == -1) {
                            endIdx = html.length()
                        }

                        def section = html.substring(startIdx, endIdx)

                        // Extract Jenkins URLs from this section
                        def sectionUrls = (section =~ /https:\/\/ci\.iotg\.sclab\.intel\.com\/job\/[^"<\s]+/).collect { it.replaceAll(/\/+$/, '') }
                        sectionUrls = sectionUrls.unique()

                        echo "============================================"
                        echo "  Daily Recompile Jobs - group_2 (${sectionUrls.size()} jobs)"
                        echo "============================================"
                        sectionUrls.eachWithIndex { url, idx ->
                            def jobName = url.tokenize('/').last()
                            echo "${idx + 1}. ${jobName}"
                            echo "   ${url}"
                        }
                        echo "============================================"
                    }
                }
            }
        }
    }
}
