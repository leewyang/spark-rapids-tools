# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core implementation class for execution of profiling tool."""

import re
from dataclasses import dataclass
from itertools import chain
from typing import List, Tuple

from tabulate import tabulate

from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.common.utilities import Utils
from spark_rapids_pytools.rapids.rapids_tool import RapidsJarTool
from spark_rapids_tools.api_v1 import ProfCore
from spark_rapids_tools.utils.data_utils import TXTResult


@dataclass
class ProfilingCore(RapidsJarTool[ProfCore]):
    """
    Core profiling tool
    """
    name = 'profiling'

    def _process_custom_args(self) -> None:
        self._process_eventlogs_args()

    def _init_rapids_arg_list(self) -> List[str]:
        rapids_threads_args = self._get_rapids_threads_count(self.name)
        autotuner_rapids_args = self._create_autotuner_rapids_args()
        return super()._init_rapids_arg_list() + ['--csv'] + rapids_threads_args + autotuner_rapids_args

    def _read_single_app_output(self, log_content: TXTResult) -> Tuple[str, List[str], List[str]]:
        """
        Read the single application profiling output log and extract the app name, recommendations and comments
        :param log_content: the content of a single application profiling output log
        :return: A tuple of (app_name, recommendations list, comments list)
        :raise Exception: if any error occurs during processing the log content (e.g. decoding issues,
                          or invalid indices)
        """
        def split_list_str_by_pattern(input_seq: List[str], pattern: str) -> int:
            ind = 0
            while ind < len(input_seq):
                if input_seq[ind].find(pattern) != -1:
                    return ind
                ind += 1
            return -1

        props_list = []
        comments_list = []
        app_name: str = ''
        raw_lines = [line.strip() for line in log_content.decode_txt().splitlines() if line.strip()]
        # find the app_name
        app_name_candidates = re.findall(r'(\|spark\.app\.name\s+\|)(.+)\|',
                                         Utils.gen_multiline_str(raw_lines),
                                         flags=re.MULTILINE)
        if len(app_name_candidates) > 0:
            _, grp_2 = app_name_candidates[0]
            app_name = grp_2.strip()
        header_pattern = self.ctxt.get_value('toolOutput', 'recommendations', 'headers',
                                             'section')
        spark_pattern = self.ctxt.get_value('toolOutput', 'recommendations', 'headers',
                                            'sparkProperties')
        comments_pattern = self.ctxt.get_value('toolOutput', 'recommendations', 'headers',
                                               'comments')
        begin_props_ind = -1
        last_props_ind = -1
        begin_comm_ind = -1
        last_comm_ind = -1
        section_ind = split_list_str_by_pattern(raw_lines, header_pattern)
        if section_ind != -1:
            recom_section = raw_lines[section_ind:]
            recom_properties_ind = split_list_str_by_pattern(recom_section,
                                                             spark_pattern)
            if recom_properties_ind not in (-1, len(recom_section) - 1):
                begin_props_ind = recom_properties_ind + 1
            recom_comments_ind = split_list_str_by_pattern(recom_section, comments_pattern)
            if recom_comments_ind != -1:
                last_props_ind = recom_comments_ind
                begin_comm_ind = recom_comments_ind + 1
                last_comm_ind = len(recom_section)
            else:
                last_props_ind = len(recom_section)
                last_comm_ind = len(recom_section)
        if begin_props_ind != -1:
            props_list = recom_section[begin_props_ind: last_props_ind]
        if begin_comm_ind != -1:
            comments_list = recom_section[begin_comm_ind: last_comm_ind]
        # Note that sorting the comments is disabled because it will change the order
        # of multiline entries
        # Recommendations can be sorted so that the two values are aligned
        # comments_list.sort()
        props_list.sort()
        return app_name, props_list, comments_list

    def _write_summary(self):
        print(Utils.gen_multiline_str(self._report_tool_full_location(),
                                      self.ctxt.get_ctxt('wrapperOutputContent')))

    def _generate_report_with_recommendations(self):
        """
        Generate a summary report containing the recommendations for each application profiled.
        The report is written to a local file and also stored in the context for STDOUT output.
        """
        log_lines = []
        recommendations_table = []
        sec_props_head = ['\tSpark Properties:']
        sec_comments_head = ['\tComments:']
        log_lines.append('### Recommended configurations ###')
        headers = self.ctxt.get_value('local', 'output', 'summaryColumns')
        with self.core_handler.txt('appRawSummaryLog') as app_rawlogs_summaries:
            # the result of the load operation is a dictionary [appid, data]
            for app_id, log_content in app_rawlogs_summaries.items():
                app_name = ''
                app_recommends = []
                app_comments = []
                try:
                    # handle exception on each single entry to avoid breaking the whole process
                    if not log_content.success:
                        self.logger.error('Failed to load [appRawSummaryLog] for App ID: %s, error: %s',
                                          app_id, log_content.get_fail_cause())
                        continue
                    app_name, app_recommends, app_comments = self._read_single_app_output(log_content)
                except Exception as e:  # pylint: disable=broad-except
                    self.logger.error('Error processing [appRawSummaryLog] for App ID: %s, error: %s',
                                      app_id, e)
                    continue
                # handle empty values
                if len(app_recommends) == 0:
                    app_recommends = ['- No recommendations']
                if len(app_comments) == 0:
                    app_comments = ['- No comments']
                if app_name == '':
                    app_name = app_id
                row = [app_id,
                       app_name,
                       Utils.gen_multiline_str(app_recommends),
                       Utils.gen_multiline_str(app_comments)]
                log_lines.append(f'App ID: {app_id}')
                sec_props = Utils.gen_joined_str(join_elem='\n\t',
                                                 items=list(chain(sec_props_head, app_recommends)))
                sec_comments = Utils.gen_joined_str(join_elem='\n\t',
                                                    items=list(chain(sec_comments_head, app_comments)))
                log_lines.append(f'{sec_props}')
                log_lines.append(f'{sec_comments}')
                recommendations_table.append(row)
        log_file_name = self.ctxt.get_value('local', 'output', 'fileName')
        summary_file = FSUtil.build_path(self.ctxt.get_csp_output_path(), log_file_name)
        self.logger.info('Writing recommendations into local file %s', summary_file)
        log_file_lines_str = Utils.gen_multiline_str(log_lines)
        with open(summary_file, 'w', encoding='utf-8') as wrapper_summary:
            wrapper_summary.write(log_file_lines_str)
        self.logger.info('Generating Full STDOUT summary report')
        # wrapper STDOUT report contains both tabular and plain text format of recommendations
        wrapper_content = [Utils.gen_report_sec_header('Recommendations'),
                           log_file_lines_str,
                           '### Recommendations Table Summary ###',
                           tabulate(recommendations_table, headers, tablefmt='grid')]
        self.ctxt.set_ctxt('wrapperOutputContent', wrapper_content)

    def _process_output(self) -> None:
        if not self._evaluate_rapids_jar_tool_output_exist():
            self.logger.warning('No output files found from profiling core tool')
        else:
            self._generate_report_with_recommendations()
            self.logger.info('Profiling core tool completed successfully')


@dataclass
class ProfilingCoreAsLocal(ProfilingCore):
    """
    ProfilingCore tool running in local mode.
    """
    description: str = 'This is the local ProfilingCore for direct JAR execution'

    def _download_remote_output_folder(self):
        self.logger.debug('Local mode skipping downloading the remote output workdir')

    def _delete_remote_dep_folder(self):
        self.logger.debug('Local mode skipping deleting the remote workdir')

    def _process_job_submission_args(self):
        self._process_local_job_submission_args()

    def _prepare_job_arguments(self):
        super()._prepare_local_job_arguments()

    def _archive_results(self):
        self._archive_local_results()
